"""ArmStats: the sole typed seam between query-layer group_summary and estimation.

The query layer emits CENTERED moments (wire format 2, from
``increment.query.builders`` and ``increment.frame``); every ddof=1
variance reduction lives on ``ArmStats``. ``ArmStats.to_summary()``
returns the canonical ``(n, mean, ddof=1 var)`` reduction for mean-type
metrics. ``ArmStats.from_raw_sums()`` adapts format-1 input through
guarded exact-centering, keeping format-1's own refuse/warn bands.

Weighted estimands go through the ``ScoreStats`` score path; this seam accepts
only unweighted canonical group summaries.
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._literals import Alternative
from increment._moment_plan import (
    SLOTS,
    X_ROLE_VARIABLES,
    X_SLOT_ROLES,
    X_UNDECLARED,
    Mask,
    Slot,
    Var,
    slot_variable,
    x_slot_variable,
)
from increment.errors import (
    CapabilityError,
    CodedModel,
    IncrementRuntimeWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation.binomial_rr import BinomialDataError

Renderer = Callable[..., str]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementRuntimeWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "estimation.armstats.centered_sum_squares_clamped",
    IncrementRuntimeWarning,
    lambda *, what, value: (
        f"{what}: centered sum of squares ({value:.6g}) is negative but "
        "within floating-point noise of zero; treating it as 0. The raw "
        "sums have cancelled (spread/|mean| near or below sqrt(eps) ~ "
        "1e-8) and retain no variance information at this scale -- "
        "centre the values (subtract a constant offset) before "
        "aggregation to recover it."
    ),
)

_register_warning(
    "estimation.armstats.centered_sum_squares_noise_floor",
    IncrementRuntimeWarning,
    lambda *, what, value, noise: (
        f"{what}: centered sum of squares ({value:.6g}) is below its "
        f"floating-point noise floor (~{noise:.3g}); the variance is "
        "unreliable at this scale (spread/|mean| near sqrt(eps) ~ 1e-8) "
        "-- centre the values (subtract a constant offset) before "
        "aggregation."
    ),
)

_register_warning(
    "estimation.armstats.centered_cross_sum_noise_floor",
    IncrementRuntimeWarning,
    lambda *, what, value, noise: (
        f"{what}: centered cross sum ({value:.6g}) is below its "
        f"floating-point noise floor (~{noise:.3g}); the covariance is "
        "unreliable at this scale (|mean| >> spread) -- centre the "
        "values (subtract a constant offset) before aggregation."
    ),
)


_EPS = sys.float_info.epsilon


def _exact_centered(sum_ab: float, sum_a: float, sum_b: float, n: int) -> tuple[float, float]:
    """Correctly rounded ``sum_ab - sum_a * sum_b / n``, via exact rational
    arithmetic on the stored floats, plus the magnitude of the two
    cancelled terms (the scale noise floors are measured against).
    """
    cross = Fraction(sum_a) * Fraction(sum_b) / n
    value = float(Fraction(sum_ab) - cross)
    magnitude = abs(sum_ab) + abs(float(cross))
    return value, magnitude


def variance_slack(magnitude: float, n: int) -> float:
    """Floating-point rounding tolerance for a centered sum of squares:
    correctly rounded sums carry up to 1/2 ulp each, and sequential
    accumulation over ``n`` addends drifts ~sqrt(n) ulps. The ONE
    tolerance formula shared by :func:`clamp_negative_variance` and
    :func:`police_sq_sum`'s refusal message.
    """
    return (4.0 + math.sqrt(n)) * _EPS * magnitude


def clamp_negative_variance(value: float, *, magnitude: float, n: int) -> float | None:
    """Magnitude-relative refuse/clamp boundary for a value that is
    mathematically a variance or sum of squares (>= 0) but may carry a
    small negative floating-point residual from catastrophic
    cancellation (see :func:`variance_slack`).

    Returns *value* unchanged when it is already non-negative, ``0.0``
    when the deficit is within the rounding tolerance, or ``None`` when
    it exceeds that tolerance -- a magnitude beyond rounding means the
    moment sums are inconsistent (mis-aggregated or corrupted
    upstream), not merely cancelled, and the caller must refuse rather
    than silently clamp.

    A non-finite *value* or *magnitude* returns ``None`` too. Neither
    comparison below is true of ``NaN``, so without this it would fall
    through and clamp to ``0.0`` -- reporting a zero standard error, i.e.
    maximal confidence, from a computation that actually overflowed.

    Pure boundary decision, no warnings/exceptions: layered on by both
    :func:`police_sq_sum` (raises, ingress convention) and the engine
    seam's own refuse-vs-clamp contract (returns a result field as
    ``None`` instead) -- the ONE tolerance decision both share.
    """
    if not math.isfinite(value) or not math.isfinite(magnitude):
        return None
    if value >= 0.0:
        return value
    if value < -variance_slack(magnitude, n):
        return None
    return 0.0


def police_sq_sum(
    value: float, magnitude: float, n: int, *, what: str, clamp_positive: bool = False
) -> float:
    """Refuse/warn policy for a centered sum of squares.

    Mathematically non-negative, but the cancelled terms agree in their
    leading digits whenever ``|mean| >> spread``: once the coefficient of
    variation nears ``sqrt(eps)`` (~1e-8), the true value falls below the
    rounding the stored sums carry and no reduction can recover it.
    """
    noise = 4.0 * _EPS * magnitude
    if value < 0.0:
        clamped = clamp_negative_variance(value, magnitude=magnitude, n=n)
        if clamped is None:
            _raise(
                "estimation.armstats.centered_sum_squares",
                what=what,
                value=value,
                variance_slack=variance_slack(magnitude, n),
            )
        _warn(
            "estimation.armstats.centered_sum_squares_clamped",
            what=what,
            value=value,
            stacklevel=3,
        )
        return 0.0
    if 0.0 < value < noise:
        _warn(
            "estimation.armstats.centered_sum_squares_noise_floor",
            what=what,
            value=value,
            noise=noise,
            stacklevel=3,
        )
        return 0.0 if clamp_positive else value
    return value


def centered_sq_sum(
    sum_sq: float, sum_: float, n: int, *, what: str, clamp_positive: bool = False
) -> float:
    """Centered sum of squares ``sum_sq - sum_**2 / n`` (``(n-1) * var``).

    Used wherever a raw sum-of-squares and its first moment must be centered
    exactly -- format-1 ingress, and any later reduction holding raw sums (the
    absorption within/between decomposition, for instance). Negative beyond
    rounding raises; negative within rounding noise returns 0.0 with a warning;
    positive but below the noise floor is returned as-is with a warning unless
    ``clamp_positive``.
    """
    value, magnitude = _exact_centered(sum_sq, sum_, sum_, n)
    return police_sq_sum(value, magnitude, n, what=what, clamp_positive=clamp_positive)


def centered_masked_sq_sum(
    sum_y2d: float, sum_yd: float, sum_d: float, ref_y: float, n: int, *, what: str
) -> float:
    """Subgroup centered sum of squares ``sum(d * (y - ref_y)**2)`` from
    format-1 raw sums, as an exact rational of the stored floats. Same
    refuse/warn policy as :func:`centered_sq_sum`.
    """
    r = Fraction(ref_y)
    exact = Fraction(sum_y2d) - 2 * r * Fraction(sum_yd) + r * r * Fraction(sum_d)
    magnitude = abs(sum_y2d) + abs(2.0 * ref_y * sum_yd) + abs(ref_y * ref_y * sum_d)
    return police_sq_sum(float(exact), magnitude, n, what=what)


def centered_cross_sum(sum_ab: float, sum_a: float, sum_b: float, n: int, *, what: str) -> float:
    """Centered cross sum ``sum_ab - sum_a * sum_b / n`` (``(n-1) * cov``).

    Format-1 input only.  Negative values are legitimate for a covariance,
    so only the noise floor is policed: a nonzero result smaller than the
    rounding carried by the cancelled terms gets a ``RuntimeWarning``.
    """
    value, magnitude = _exact_centered(sum_ab, sum_a, sum_b, n)
    noise = 4.0 * _EPS * magnitude
    if 0.0 < abs(value) < noise:
        _warn(
            "estimation.armstats.centered_cross_sum_noise_floor",
            what=what,
            value=value,
            noise=noise,
            stacklevel=3,
        )
    return value


def scaled_cross_over_n(a: float, b: float, n: int) -> float:
    """``a * b / n``, computed by descaling each factor by ``sqrt(n)``
    first so the numerator product never overflows float64 even when
    the true quotient is representable (e.g. two residual first
    moments near the float64 max with a moderate ``n``: ``a * b``
    alone overflows to ``inf`` long before dividing by ``n`` would
    bring it back down). Exact in real arithmetic; the same order of
    floating-point rounding as the direct ``a * b / n`` form otherwise.

    The shared "overflow-safe cross term": every ``ddof=1`` covariance
    reduction and every ``from_raw_sums`` bias-correction term on this
    class is a first-moment product divided by ``n``.
    """
    root_n = math.sqrt(n)
    return (a / root_n) * (b / root_n)


def assert_cauchy_schwarz(cross: float, var_a: float, var_b: float, n: int, *, what: str) -> None:
    """Refuse a cross moment inconsistent with Cauchy-Schwarz: any two
    real vectors sharing ``n`` addends satisfy ``cross**2 <= var_a *
    var_b`` exactly, before any floating-point rounding at all -- e.g.
    a covariance against its two variances, or a residual first moment
    (``cy1``) against its own second moment and ``n`` (the constant-1
    vector's "variance" is ``n``). A violation beyond rounding means
    the moments were not produced by the same ``n`` values: mismatched
    partitions, mis-aggregation, or corruption -- not a floating-point
    cancellation, since this bound holds prior to any rounding.

    Computed as ``sqrt(var_a) * sqrt(var_b)`` rather than
    ``sqrt(var_a * var_b)``, so a legitimately huge magnitude (e.g.
    ``var_b ~ 1e308``) never overflows the comparison itself (``sqrt``
    of any finite float64 is always representable).
    Silently accepts a negative variance: a genuine sum of squares can't
    be negative (:func:`police_sq_sum`'s refusal to make, not this
    check's). An EXACTLY zero variance makes the exact bound zero, so the
    tolerance becomes absolute rather than relative: a denormal-scale
    artifact from an upstream reduction is still accepted, but a
    materially nonzero cross moment against a zero variance cannot come
    from any real sample and is refused.
    """
    if var_a < 0.0 or var_b < 0.0:
        return
    if var_a == 0.0 or var_b == 0.0:
        floor = variance_slack(max(abs(var_a), abs(var_b)), n)
        if abs(cross) <= floor:
            return
        _raise(
            "estimation.armstats.cross_moment_materially",
            what=what,
            cross=cross,
            var_a=var_a,
            var_b=var_b,
            floor=floor,
        )
    bound = math.sqrt(var_a) * math.sqrt(var_b)
    excess = abs(cross) - bound
    if excess <= 0.0 or excess <= variance_slack(bound, n):
        return
    _raise(
        "estimation.armstats.cross_moment_violates",
        what=what,
        cross=cross,
        var_a=var_a,
        var_b=var_b,
        bound=bound,
        excess=excess,
        variance_slack=variance_slack(bound, n),
    )


def welch_satterthwaite_df(var_a: float, dof_a: float, var_b: float, dof_b: float) -> float:
    """Welch-Satterthwaite effective degrees of freedom for the sum of two
    INDEPENDENT variance components: ``var_a`` (estimated with ``dof_a``
    degrees of freedom) plus ``var_b`` (``dof_b`` degrees)::

        nu = (var_a + var_b)**2 / (var_a**2 / dof_a + var_b**2 / dof_b)

    The effective df equals dof_a+dof_b when var_a/dof_a equals
    var_b/dof_b; equal components alone are insufficient if the dfs differ.
    This is a moment-matched reference, generally not an exact Student law.

    Normalize by the larger variance before squaring. The formula is
    scale-invariant (both numerator and denominator are degree two in the
    variances). Both-zero components return dof_a as a deterministic fallback;
    a caller must still handle the unavailable zero-variance interval.
    """
    scale = max(var_a, var_b)
    if scale <= 0.0:
        return dof_a
    a, b = var_a / scale, var_b / scale
    denom = a * a / dof_a + b * b / dof_b
    return (a + b) ** 2 / denom if denom > 0.0 else dof_a


def _pooled_reference(parts: Sequence[CenteredMoments], variable: Var, n: int) -> float:
    """Count-weighted mean of *variable*'s reference across partitions, anchored on the first.

    Anchoring keeps the accumulator on the between-partition spread instead
    of on ``sum(v)`` itself, so no large magnitude is ever summed.
    """
    base = parts[0].ref[variable]
    excess = 0.0
    for part in parts:
        excess += part.n * (part.ref[variable] - base) + part.c1[(variable, None)]
    return base + excess / n


@dataclass(frozen=True, slots=True)
class CenteredMoments:
    """Centered, mergeable moments of named variables, optionally masked.

    ``ref[v]`` is *v*'s own partition mean. ``c1[(v, m)]`` is
    ``sum(m * (v - ref_v))`` -- ``m=None`` is the all-ones mask, so the
    unmasked ``c1`` is the "~0 but exact" residual and ``sum(v) == n*ref_v
    + c1`` recovers exactly. ``c2[(a, b, m)]`` is ``sum(m * (a - ref_a) *
    (b - ref_b))``, keyed with ``(a, b)`` in declaration order.
    ``count[m]`` is ``sum(m)``, with ``count[None] == n``. A masked moment
    stays centered on the PARENT reference, never on the masked subgroup's
    own mean: the between-subgroup mean difference is then a per-row exact
    residual, not a difference of two large means.

    Combination across partitions is closed here (:meth:`combine`). With
    ``delta = ref_part - ref_pooled`` per variable::

        c1     += c1_p   + count_p * delta
        c2(v,v) += c2_p  + 2.0 * delta * c1_p(v) + count_p * delta * delta
        c2(a,b) += c2_p  + delta_a * c1_p(b) + delta_b * c1_p(a) + count_p * delta_a * delta_b

    Every term is ``O(delta**2)`` in the between-partition spread. The
    diagonal is written ``2.0 * delta * c1``, never ``delta * c1`` added
    twice: the two round differently.

    ``successes`` is the exact number of units whose binary outcome ``y`` is 1, an integer kept
    apart from the floating slots: ``None`` unless a producer declared ``y`` a 0/1 outcome of
    each of the ``n`` units. It describes the outcome ``y`` itself, so it follows every
    operation that leaves ``y`` untouched (renaming, aliasing or dropping other variables) and
    is dropped by one that replaces or removes ``y``. Partitions add their counts as integers;
    a partition without one leaves the pooled record without.
    """

    n: int
    variables: tuple[Var, ...]
    ref: Mapping[Var, float]
    c1: Mapping[tuple[Var, Mask], float]
    c2: Mapping[tuple[Var, Var, Mask], float]
    count: Mapping[Mask, float]
    successes: int | None = None

    def __post_init__(self) -> None:
        order = {variable: index for index, variable in enumerate(self.variables)}
        c2: dict[tuple[Var, Var, Mask], float] = {}
        for (a, b, mask), value in self.c2.items():
            c2[(a, b, mask) if order[a] <= order[b] else (b, a, mask)] = value
        count: dict[Mask, float] = {None: float(self.n)}
        count.update((mask, value) for mask, value in self.count.items() if mask is not None)
        object.__setattr__(self, "ref", MappingProxyType(dict(self.ref)))
        object.__setattr__(self, "c1", MappingProxyType(dict(self.c1)))
        object.__setattr__(self, "c2", MappingProxyType(c2))
        object.__setattr__(self, "count", MappingProxyType(count))

    def pair(self, a: Var, b: Var) -> tuple[Var, Var]:
        """``(a, b)`` in declaration order, the key order of ``c2``."""
        return (a, b) if self.variables.index(a) <= self.variables.index(b) else (b, a)

    def has(self, variable: Var) -> bool:
        return variable in self.ref

    # First moments: exact recovery from reference + residual

    def mean(self, variable: Var) -> float:
        return self.ref[variable] + self.c1[(variable, None)] / self.n

    def mask_mean(self, mask: str) -> float:
        return self.count[mask] / self.n

    def masked_mean(self, variable: Var, mask: str) -> float:
        """Mean of ``m * v``: a product, never a difference of large numbers."""
        return (self.c1[(variable, mask)] + self.ref[variable] * self.count[mask]) / self.n

    def unmasked_mean(self, variable: Var, mask: str) -> float:
        """Mean of ``(1 - m) * v``, expanded so no two large means are subtracted."""
        return (
            self.ref[variable] * (self.n - self.count[mask])
            + self.c1[(variable, None)]
            - self.c1[(variable, mask)]
        ) / self.n

    # ddof=1 reductions: every / (n-1) lives here

    def var(self, variable: Var, *, what: str) -> float:
        """``sum((v - vbar)**2) / (n - 1)``: ``c2 - c1**2/n`` exactly, ``c1`` ~0.

        A negative ``c2`` cannot come from a sum of squares: corrupt input,
        refused by name. ``c1**2/n`` goes through :func:`scaled_cross_over_n`
        so a huge but legitimate reference offset cannot overflow it.
        """
        c2 = self.c2[(variable, variable, None)]
        c1 = self.c1[(variable, None)]
        if c2 < 0.0:
            _raise("estimation.armstats.arm_stats.centered_sum_squares", what=what, c2=c2)
        return max(c2 - scaled_cross_over_n(c1, c1, self.n), 0.0) / (self.n - 1)

    def cov(self, a: Var, b: Var) -> float:
        cross = self.c2[(*self.pair(a, b), None)]
        return (cross - scaled_cross_over_n(self.c1[(a, None)], self.c1[(b, None)], self.n)) / (
            self.n - 1
        )

    def mask_var(self, mask: str) -> float:
        """Binary mask variance ``k*(n-k)/n/(n-1)``: exact integer products, no rounding."""
        k = self.count[mask]
        return k * (self.n - k) / self.n / (self.n - 1)

    def cov_mask(self, variable: Var, mask: str) -> float:
        """``Cov(v, m)``: ``sum(v*m) - sum(v)*sum(m)/n`` collapses to ``c1[v|m] - c1[v]*k/n``."""
        return (
            self.c1[(variable, mask)]
            - scaled_cross_over_n(self.c1[(variable, None)], self.count[mask], self.n)
        ) / (self.n - 1)

    def masked_product_var(self, variable: Var, mask: str, *, what: str) -> float:
        """ddof=1 variance of the product ``m * v``::

            c2[v,v|m] + 2*ref*c1[v|m]*(1-p) + ref**2*k*(1-p) - c1[v|m]**2/n

        with ``p = k/n``; the third term is the real between-group component.
        """
        n, k, r = self.n, self.count[mask], self.ref[variable]
        c1m = self.c1[(variable, mask)]
        c2m = self.c2[(variable, variable, mask)]
        q = 1.0 - k / n
        term2 = 2.0 * r * c1m * q
        term3 = r * r * k * q
        term4 = c1m * c1m / n
        value = c2m + term2 + term3 - term4
        magnitude = abs(c2m) + abs(term2) + abs(term3) + abs(term4)
        centered = police_sq_sum(value, magnitude, n, what=what, clamp_positive=True)
        return centered / (n - 1)

    def cov_masked_product(self, variable: Var, mask: str) -> float:
        """ddof=1 ``Cov(v, m*v)``: ``c2[v,v|m] + ref*(c1[v|m] - c1[v]*k/n) - c1[v]*c1[v|m]/n``."""
        n, k, r = self.n, self.count[mask], self.ref[variable]
        c1 = self.c1[(variable, None)]
        c1m = self.c1[(variable, mask)]
        c2m = self.c2[(variable, variable, mask)]
        return (c2m + r * (c1m - c1 * k / n) - c1 * c1m / n) / (n - 1)

    # Renaming

    def renamed(self, names: Mapping[Var, Var]) -> CenteredMoments:
        """The same moments with variables renamed per *names* (old -> new)."""

        def new(variable: Var) -> Var:
            return names.get(variable, variable)

        outcome_kept = new("y") == "y" and all(
            new(variable) != "y" for variable in self.variables if variable != "y"
        )
        return CenteredMoments(
            n=self.n,
            variables=tuple(new(v) for v in self.variables),
            ref={new(v): value for v, value in self.ref.items()},
            c1={(new(v), mask): value for (v, mask), value in self.c1.items()},
            c2={(new(a), new(b), mask): value for (a, b, mask), value in self.c2.items()},
            count=self.count,
            successes=self.successes if outcome_kept else None,
        )

    def with_alias(self, alias: Var, *, of: Var) -> CenteredMoments:
        """*alias* carries *of*'s moments under a second name, cross terms included."""
        c1 = dict(self.c1)
        for (variable, mask), value in self.c1.items():
            if variable == of:
                c1[(alias, mask)] = value
        c2 = dict(self.c2)
        for (a, b, mask), value in self.c2.items():
            for one, other in ((a, b), (b, a)):
                if one == of:
                    c2[(other, alias, mask)] = value
            if a == of and b == of:
                c2[(alias, alias, mask)] = value
        return CenteredMoments(
            n=self.n,
            variables=(*self.variables, alias),
            ref={**self.ref, alias: self.ref[of]},
            c1=c1,
            c2=c2,
            count=self.count,
            successes=self.successes if alias != "y" else None,
        )

    def without(self, variable: Var) -> CenteredMoments:
        """The same moments with *variable* and every moment mentioning it removed."""
        return CenteredMoments(
            n=self.n,
            variables=tuple(v for v in self.variables if v != variable),
            ref={v: value for v, value in self.ref.items() if v != variable},
            c1={key: value for key, value in self.c1.items() if key[0] != variable},
            c2={key: value for key, value in self.c2.items() if variable not in key[:2]},
            count=self.count,
            successes=self.successes if variable != "y" else None,
        )

    # Combination across partitions

    @classmethod
    def combine(cls, parts: Sequence[CenteredMoments]) -> CenteredMoments:
        """Reduce partitions of one population to one record (the shift law above).

        Every part must declare the same variables and masks; a first or
        second moment present on only some parts is dropped. The pooled
        reference is anchored on the first part so no large magnitude is
        ever summed; the result is associative and order-independent to
        floating-point roundoff.
        """
        head = parts[0]
        assert all(part.variables == head.variables for part in parts), (
            "parts must share one declaration order"
        )
        n = sum(part.n for part in parts)
        successes = (
            sum(part.successes for part in parts if part.successes is not None)
            if all(part.successes is not None for part in parts)
            else None
        )
        ref = {variable: _pooled_reference(parts, variable, n) for variable in head.variables}
        c1_keys = [key for key in head.c1 if all(key in part.c1 for part in parts)]
        c2_keys = [key for key in head.c2 if all(key in part.c2 for part in parts)]
        masks = [mask for mask in head.count if mask is not None]
        c1 = dict.fromkeys(c1_keys, 0.0)
        c2 = dict.fromkeys(c2_keys, 0.0)
        count: dict[Mask, float] = dict.fromkeys(masks, 0.0)
        for part in parts:
            delta = {variable: part.ref[variable] - ref[variable] for variable in head.variables}
            for variable, mask in c1_keys:
                c1[(variable, mask)] += (
                    part.c1[(variable, mask)] + part.count[mask] * delta[variable]
                )
            for a, b, mask in c2_keys:
                if a == b:
                    c2[(a, b, mask)] += (
                        part.c2[(a, b, mask)]
                        + 2.0 * delta[a] * part.c1[(a, mask)]
                        + part.count[mask] * delta[a] * delta[a]
                    )
                else:
                    c2[(a, b, mask)] += (
                        part.c2[(a, b, mask)]
                        + delta[a] * part.c1[(b, mask)]
                        + delta[b] * part.c1[(a, mask)]
                        + part.count[mask] * delta[a] * delta[b]
                    )
            for mask in masks:
                count[mask] += part.count[mask]
        return cls(
            n=n, variables=head.variables, ref=ref, c1=c1, c2=c2, count=count, successes=successes
        )


def _slot_operands(by_slot: Mapping[str, Var | None], slot: Slot) -> tuple[Var, ...] | None:
    """Plan variables filling *slot*'s wire variables, or None when one is unfilled."""
    operands: list[Var] = []
    for wire_variable in slot.variables:
        variable = by_slot[wire_variable]
        if variable is None:
            return None
        operands.append(variable)
    return tuple(operands)


def slot_fields(moments: CenteredMoments) -> dict[str, Any]:
    """The sixteen format-8 slots plus ``x_role`` for *moments*; an unfilled slot is None."""
    x_candidates = [v for v in moments.variables if slot_variable(v) == "x"]
    assert len(x_candidates) <= 1, x_candidates
    x_var = x_candidates[0] if x_candidates else None
    by_slot: dict[str, Var | None] = {"y": "y", "x": x_var, "den": "den"}
    fields: dict[str, Any] = {}
    for column, slot in SLOTS.items():
        names = _slot_operands(by_slot, slot)
        if names is None or not all(moments.has(name) for name in names):
            fields[column] = None
        elif slot.kind == "ref":
            fields[column] = moments.ref[names[0]]
        elif slot.kind == "c1":
            fields[column] = moments.c1.get((names[0], slot.mask))
        elif slot.kind == "c2":
            fields[column] = moments.c2.get((*moments.pair(names[0], names[1]), slot.mask))
        else:
            fields[column] = moments.count.get(slot.mask)
    fields["x_role"] = X_SLOT_ROLES.get(x_var) if x_var is not None else None
    return fields


class SummaryStats(CodedModel, BaseModel):
    """Canonical reduction of one arm's raw moments.

    ``var`` is ALWAYS ddof=1 (unbiased variance of the unit-level values).
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=1)  # a summary exists only for a non-empty arm
    mean: float
    var: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> SummaryStats:
        if not math.isfinite(self.mean):
            _raise("estimation.armstats.summary_stats.mean_finite", mean=self.mean)
        if not math.isfinite(self.var):
            _raise("estimation.armstats.summary_stats.var_finite", var=self.var)
        return self


#: Wire columns of the centered (format-2) moment families, in emission
#: order, plus the ``x_role`` declaration that rides alongside the x family.
CENTERED_FIELDS = (*SLOTS, "x_role")

#: Wire columns of the format-1 (raw additive sums) moment families.
RAW_SUM_FIELDS = (
    "sum_y",
    "sum_y2",
    "sum_x",
    "sum_x2",
    "sum_xy",
    "sum_den",
    "sum_den2",
    "sum_yden",
    "sum_d",
    "sum_yd",
    "sum_y2d",
    "sum_xd",
)


class ArmStats(CodedModel, BaseModel):
    """One arm's aggregated record (per metric x group), stored as CENTERED
    moments so no field carries an ``O(n * mean**2)`` term a reduction must
    subtract back off.

    This is the sole typed seam: the query layer emits ``group_summary`` rows
    as a DataFrame, validated into ``list[ArmStats]`` once at the edge.

    * ``ref_y``/``ref_x``/``ref_den``: each family's own partition mean
      (``None`` when the family is absent).
    * ``cy1``/``cx1``/``cden1``: residual first moments (``sum(y - ref_y)``),
      ~0 but kept exact so ``sum(y) == n*ref_y + cy1`` recovers exactly.
    * ``cy2``/``cx2``/``cxy``/``cden2``/``cyden``: centered second and cross
      moments, e.g. ``cxy = sum((x - ref_x) * (y - ref_y))``.
    * ``cyd``/``cy2d``/``cxd``: the uptake-masked family, centered on the
      OVERALL partition reference, e.g. ``cyd = sum(d * (y - ref_y))``.

    ``n`` is an exact integer and ``sum_d`` a float sum, exact to 2**53 units.

    ``successes`` is the exact number of units with ``y == 1``, set by a producer that declared
    ``y`` a 0/1 outcome of each of the ``n`` units and ``None`` elsewhere (adjusted, ratio,
    clustered, transformed or non-binary outcomes). Floating moments cannot identify a count
    once an arm nears ``2**53`` units, so it travels as an integer beside them, and the
    conversion route (:func:`binary_counts`) reads it as the count, never the moments.

    **``cy2d / (n - 1)`` is NOT ``Var(d*y)``**: the subgroup fields carry
    only WITHIN-subgroup dispersion; the between-group term, ``ref_y**2 *
    sum_d * (1 - p)`` (``p = sum_d/n``), enters through :meth:`var_yd`, so
    every reduction here expands analytically in the centered fields.

    Combination across partitions (day slices, breakout cells, pooled arms)
    is closed in this space - see :meth:`combine`. With
    ``delta_p = ref_p - R`` for a target reference ``R``::

        S(y-R)^2   = sum_p [ cy2_p + 2*delta_p*cy1_p + n_p*delta_p^2 ]
        S d(y-R)^2 = sum_p [ cy2d_p + 2*delta_p*cyd_p + sum_d_p*delta_p^2 ]
        S d(y-R)   = sum_p [ cyd_p + sum_d_p*delta_p ]
        S (x-Rx)(y-Ry) = sum_p [ cxy_p + delta_x_p*cy1_p + delta_y_p*cx1_p
                                 + n_p*delta_x_p*delta_y_p ]

    Every term is ``O(delta**2)`` in the between-partition mean spread, so
    combination is associative and order-independent to fp roundoff.

    The ``x`` family is ``None`` until CUPED exists; ``den`` is ``None`` for
    non-ratio metrics; ``cxd`` needs both a covariate and an uptake fact.
    """

    model_config = ConfigDict(frozen=True)

    study_id: str
    metric: str
    group_id: str
    n: int = Field(ge=1)  # a group_summary row exists only for a non-empty arm
    # Exact units with y == 1 for a declared 0/1 outcome of every unit; None elsewhere.
    successes: int | None = Field(default=None, strict=True)
    ref_y: float
    cy1: float
    cy2: float
    ref_x: float | None = None
    cx1: float | None = None
    cx2: float | None = None
    cxy: float | None = None  # None until CUPED covariate materialised
    # Role of the x family: 'covariate' | 'cluster_size' | 'uptake_total';
    # None when the x family is absent (x_role is set iff ref_x is set).
    x_role: str | None = None
    ref_den: float | None = None
    cden1: float | None = None
    cden2: float | None = None
    cyden: float | None = None
    cxden: float | None = None  # cov(x, den); the clustered LATE collapse
    # repurposes x to carry each cluster's uptake total, so this carries
    # cov(cluster uptake total, cluster size) - see cov_xden()
    sum_d: float | None = None  # binary uptake; None = not an encouragement analysis
    cyd: float | None = None
    cy2d: float | None = None  # powers the complier-relative SE
    cxd: float | None = None  # needed only by CUPED-adjusted LATE
    winsor_lower_percentile: float | None = None
    winsor_upper_percentile: float | None = None
    winsor_lower_bound: float | None = None
    winsor_upper_bound: float | None = None
    winsor_n: int | None = Field(default=None, ge=1)
    winsor_n_lower: int | None = Field(default=None, ge=0)
    winsor_n_upper: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _reject_obsolete_weight_fields(cls, values: Any) -> Any:
        if isinstance(values, Mapping):
            obsolete = sorted({"sum_w", "sum_w2"}.intersection(values))
            if obsolete:
                _raise("estimation.armstats.arm_stats.obsolete_armstats_fields", obsolete=obsolete)
        return values

    @model_validator(mode="after")
    def _validate_centered_moments(self) -> ArmStats:
        """Validate the canonical centered-moment ingress invariants."""
        core = ("n", "ref_y", "cy1", "cy2")
        for name in core:
            value = getattr(self, name)
            if not math.isfinite(value):
                _raise(
                    "estimation.armstats.arm_stats.group_summary_moment",
                    name=name,
                    group_id=self.group_id,
                    metric=self.metric,
                    value=value,
                )

        if self.successes is not None and not 0 <= self.successes <= self.n:
            _raise(
                "estimation.binomial.reconstructed_counts_not_binary",
                group_id=self.group_id,
                metric=self.metric,
                n=self.n,
                sum_y=self.successes,
            )

        families = (
            ("covariate", ("ref_x", "cx1", "cx2", "cxy")),
            ("denominator", ("ref_den", "cden1", "cden2", "cyden")),
            ("uptake", ("sum_d", "cyd", "cy2d")),
        )
        for label, fields in families:
            present = [name for name in fields if getattr(self, name) is not None]
            if present and len(present) != len(fields):
                missing = [name for name in fields if name not in present]
                _raise(
                    "estimation.armstats.arm_stats.partial_family_missing",
                    label=label,
                    missing=missing,
                    present=present,
                )
            for name in fields:
                value = getattr(self, name)
                if value is not None and not math.isfinite(value):
                    _raise(
                        "estimation.armstats.arm_stats.cross_field_finite",
                        name=name,
                        value=value,
                    )

        for name in ("cxden", "cxd"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                _raise("estimation.armstats.arm_stats.cross_field_finite", name=name, value=value)

        has_x = self.ref_x is not None
        if self.x_role is not None and self.x_role not in X_ROLE_VARIABLES:
            _raise("estimation.armstats.arm_stats.x_role_one", x_role=self.x_role)
        # Rows with x and cxden but no x_role have ambiguous clustered semantics.
        # Preserve that state so the cluster reducer can issue its specific refusal.
        if has_x and self.x_role is None and self.cxden is None:
            _raise("estimation.armstats.arm_stats.x_role_declared")
        if not has_x and self.x_role is not None:
            _raise("estimation.armstats.arm_stats.x_role_family")

        has_den = self.ref_den is not None
        has_uptake = self.sum_d is not None
        if self.cxden is not None and not (has_x and has_den):
            _raise("estimation.armstats.arm_stats.cxden_complete_denominator")
        if self.cxd is not None and not (has_x and has_uptake):
            _raise("estimation.armstats.arm_stats.cxd_complete_uptake")
        if has_uptake:
            assert self.sum_d is not None
            if not (0.0 <= self.sum_d <= self.n):
                _raise(
                    "estimation.armstats.arm_stats.sum_d_outside",
                    group_id=self.group_id,
                    metric=self.metric,
                    n=self.n,
                    sum_d=self.sum_d,
                )
            # Accept fractional sum_d: analytic inputs use n*p, and the same
            # variance identity remains valid.
            # With no uptake, every uptake-masked moment must be zero; otherwise
            # the mask and moments describe different units.
            if self.sum_d == 0.0:
                masked = {"cyd": self.cyd, "cy2d": self.cy2d, "cxd": self.cxd}
                nonzero = sorted(k for k, v in masked.items() if v not in (None, 0.0))
                if nonzero:
                    _raise(
                        "estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero",
                        group_id=self.group_id,
                        metric=self.metric,
                        nonzero=nonzero,
                    )

        # Each first moment is a cross moment with the constant 1, so cy1**2 <= n * cy2 (and
        # likewise for x and den) holds before rounding; a violation means the stored reference
        # is not this arm's mean. Cross-family pairs are checked where they are reduced, with
        # the magnitude context needed to accept valid catastrophic-cancellation inputs.
        label = f"metric={self.metric!r} group={self.group_id!r}"
        assert_cauchy_schwarz(
            self.cy1, float(self.n), self.cy2, self.n, what=f"cy1 vs cy2 for {label}"
        )
        if has_x:
            assert self.cx1 is not None and self.cx2 is not None
            assert_cauchy_schwarz(
                self.cx1, float(self.n), self.cx2, self.n, what=f"cx1 vs cx2 for {label}"
            )
        if has_den:
            assert self.cden1 is not None and self.cden2 is not None
            assert_cauchy_schwarz(
                self.cden1, float(self.n), self.cden2, self.n, what=f"cden1 vs cden2 for {label}"
            )
        return self

    @model_validator(mode="after")
    def _winsor_metadata_is_complete(self) -> ArmStats:
        metadata = (
            self.winsor_lower_percentile,
            self.winsor_upper_percentile,
            self.winsor_lower_bound,
            self.winsor_upper_bound,
            self.winsor_n,
            self.winsor_n_lower,
            self.winsor_n_upper,
        )
        if all(value is None for value in metadata):
            return self
        if self.winsor_n is None or self.winsor_n_lower is None or self.winsor_n_upper is None:
            _raise("estimation.armstats.arm_stats.winsorization_metadata_winsor")
        if (
            self.winsor_n_lower > self.winsor_n
            or self.winsor_n_upper > self.winsor_n
            or self.winsor_n_lower + self.winsor_n_upper > self.winsor_n
        ):
            _raise("estimation.armstats.arm_stats.winsorization_cap_counts")
        for name in (
            "winsor_lower_percentile",
            "winsor_upper_percentile",
            "winsor_lower_bound",
            "winsor_upper_bound",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                _raise("estimation.armstats.arm_stats.winsorization_metadata_finite", name=name)
        if self.winsor_lower_percentile is not None and self.winsor_lower_bound is None:
            _raise("estimation.armstats.arm_stats.winsorization_lower_percentile")
        if self.winsor_upper_percentile is not None and self.winsor_upper_bound is None:
            _raise("estimation.armstats.arm_stats.winsorization_upper_percentile")
        if (
            self.winsor_lower_bound is not None
            and self.winsor_upper_bound is not None
            and self.winsor_lower_bound >= self.winsor_upper_bound
        ):
            _raise("estimation.armstats.arm_stats.winsorization_resolved_lower")
        return self

    # The algebra view

    @property
    def x_variable(self) -> Var | None:
        """Plan variable the x slot carries -- ``x``, ``size``, ``uptake``, or
        ``X_UNDECLARED`` for a clustered row predating the declaration; None
        when the x family is absent."""
        return x_slot_variable(self.x_role) if self.ref_x is not None else None

    @property
    def moments(self) -> CenteredMoments:
        """This row's moments as an algebra, rebuilt from the wire fields on each
        access (never cached: ``model_copy`` would carry a stale cache). The x
        slot is declared first so cross terms expand in the format-8 operand order."""
        x_var = self.x_variable
        by_slot: dict[str, Var | None] = {
            "y": "y",
            "x": x_var,
            "den": "den" if self.ref_den is not None else None,
        }
        variables = tuple(v for v in (x_var, "y", by_slot["den"]) if v is not None)
        ref: dict[Var, float] = {}
        c1: dict[tuple[Var, Mask], float] = {}
        c2: dict[tuple[Var, Var, Mask], float] = {}
        count: dict[Mask, float] = {}
        for column, slot in SLOTS.items():
            value = getattr(self, column)
            if value is None:
                continue
            names = _slot_operands(by_slot, slot)
            assert names is not None, column  # validated families
            if slot.kind == "ref":
                ref[names[0]] = value
            elif slot.kind == "c1":
                c1[(names[0], slot.mask)] = value
            elif slot.kind == "c2":
                c2[(names[0], names[1], slot.mask)] = value
            else:
                count[slot.mask] = value
        return CenteredMoments(
            n=self.n,
            variables=variables,
            ref=ref,
            c1=c1,
            c2=c2,
            count=count,
            successes=self.successes,
        )

    @classmethod
    def from_moments(
        cls,
        moments: CenteredMoments,
        *,
        study_id: str,
        metric: str,
        group_id: str,
        **winsor: Any,
    ) -> ArmStats:
        """Write *moments* into the format-8 slots (see ``_moment_plan.SLOTS``)."""
        return cls(
            study_id=study_id,
            metric=metric,
            group_id=group_id,
            n=moments.n,
            **slot_fields(moments),
            successes=moments.successes,
            **winsor,
        )

    # Format-1 adapter

    @classmethod
    # Public raw-sum adapter signature is the API for stored evidence.
    def from_raw_sums(  # noqa: PLR0913
        cls,
        *,
        study_id: str,
        metric: str,
        group_id: str,
        n: int,
        sum_y: float,
        sum_y2: float,
        sum_x: float | None = None,
        sum_x2: float | None = None,
        sum_xy: float | None = None,
        sum_den: float | None = None,
        sum_den2: float | None = None,
        sum_yden: float | None = None,
        sum_d: float | None = None,
        sum_yd: float | None = None,
        sum_y2d: float | None = None,
        sum_xd: float | None = None,
        successes: int | None = None,
    ) -> ArmStats:
        """Adapt one format-1 (raw additive sums) record to this seam,
        through the same guarded exact-centering and refuse/warn bands the
        raw-sum reductions applied before the format bump - the guard now
        fires where data enters rather than deep inside a downstream
        estimator.

        Raises ``ValueError`` if a centered sum of squares is negative
        beyond floating-point rounding; warns ``RuntimeWarning`` when a
        centered moment falls below the noise floor of the raw sums it
        was recovered from.

        ``successes`` is the exact count of ones when the caller knows ``y`` is a 0/1 outcome of
        all ``n`` units and holds that integer from its own inputs (a raw sum of floats cannot
        establish it past ``2**53``, and is never read as one here). It is stamped as given: the
        residual ``cy1`` stays the source float sum's own, so a sum that has drifted from the
        count keeps its drift, and the stored moments are checked against the count where the
        conversion route reads it (:func:`binary_counts`).
        """
        label = f"metric={metric!r} group={group_id!r}"
        ref_y = sum_y / n
        cy1 = float(Fraction(sum_y) - n * Fraction(ref_y))
        cy2 = centered_sq_sum(
            sum_y2,
            sum_y,
            n,
            what=f"var_y for {label}",
            clamp_positive=True,
        ) + scaled_cross_over_n(cy1, cy1, n)

        ref_x = cx1 = cx2 = cxy = None
        if sum_x is not None and sum_x2 is not None and sum_xy is not None:
            ref_x = sum_x / n
            cx1 = float(Fraction(sum_x) - n * Fraction(ref_x))
            cx2 = centered_sq_sum(
                sum_x2, sum_x, n, what=f"var_x for {label}"
            ) + scaled_cross_over_n(cx1, cx1, n)
            cxy = centered_cross_sum(
                sum_xy, sum_y, sum_x, n, what=f"cov_yx for {label}"
            ) + scaled_cross_over_n(cx1, cy1, n)

        ref_den = cden1 = cden2 = cyden = None
        if sum_den is not None and sum_den2 is not None and sum_yden is not None:
            ref_den = sum_den / n
            cden1 = float(Fraction(sum_den) - n * Fraction(ref_den))
            cden2 = centered_sq_sum(
                sum_den2, sum_den, n, what=f"var_den for {label}"
            ) + scaled_cross_over_n(cden1, cden1, n)
            cyden = centered_cross_sum(
                sum_yden, sum_y, sum_den, n, what=f"cov_yden for {label}"
            ) + scaled_cross_over_n(cy1, cden1, n)

        cyd = cy2d = cxd = None
        if sum_d is not None and sum_yd is not None and sum_y2d is not None:
            cyd = centered_cross_sum(
                sum_yd, sum_y, sum_d, n, what=f"cov_yd for {label}"
            ) + scaled_cross_over_n(cy1, sum_d, n)
            cy2d = centered_masked_sq_sum(
                sum_y2d, sum_yd, sum_d, ref_y, n, what=f"var_yd for {label}"
            )
        if sum_xd is not None and sum_x is not None and sum_d is not None and cx1 is not None:
            cxd = centered_cross_sum(
                sum_xd, sum_x, sum_d, n, what=f"cov_xd for {label}"
            ) + scaled_cross_over_n(cx1, sum_d, n)

        return cls(
            study_id=study_id,
            metric=metric,
            group_id=group_id,
            n=n,
            ref_y=ref_y,
            cy1=cy1,
            cy2=cy2,
            ref_x=ref_x,
            # A format-1 covariate is always a CUPED covariate; ref_x is set
            # iff one was supplied, so this preserves x_role-set-iff-ref_x-set.
            x_role="covariate" if ref_x is not None else None,
            cx1=cx1,
            cx2=cx2,
            cxy=cxy,
            ref_den=ref_den,
            cden1=cden1,
            cden2=cden2,
            cyden=cyden,
            sum_d=sum_d,
            cyd=cyd,
            cy2d=cy2d,
            cxd=cxd,
            successes=successes,
        )

    # Combination across partitions

    @classmethod
    def combine(cls, arms: Sequence[ArmStats], *, group_id: str | None = None) -> ArmStats:
        """Reduce partitions of the same population (day slices, breakout
        cells, or arms pooled for a CUPED theta) to one centered record,
        using the shift law on :class:`CenteredMoments`. The pooled
        reference is anchored on the first partition so no large magnitude
        is ever summed; combination is associative and order-independent
        to fp roundoff (Chan/Welford generalized to cross- and
        subgroup-moments).

        ``group_id=None`` requires every partition to agree on
        ``group_id``. Raises ``ValueError`` on an empty sequence,
        disagreeing study/metric, a family present on some partitions and
        absent on others, or a disagreeing ``group_id`` with no explicit
        label.
        """
        if not arms:
            _raise("estimation.armstats.arm_stats.combine_needs_least")
        head = arms[0]
        for a in arms:
            if a.study_id != head.study_id or a.metric != head.metric:
                _raise(
                    "estimation.armstats.arm_stats.combine_needs_one",
                    a_metric=a.metric,
                    a_study_id=a.study_id,
                    head_metric=head.metric,
                    head_study_id=head.study_id,
                )
        if group_id is None:
            labels = {a.group_id for a in arms}
            if len(labels) != 1:
                _raise(
                    "estimation.armstats.arm_stats.combine_partitions_from",
                    labels=sorted(labels),
                )
            group_id = head.group_id

        for family, probe in (("covariate", "ref_x"), ("denominator", "ref_den")):
            present = {getattr(a, probe) is not None for a in arms}
            if len(present) != 1:
                _raise(
                    "estimation.armstats.arm_stats.combine_family_some",
                    family=family,
                    probe=probe,
                )
        uptake = {a.sum_d is not None for a in arms}
        if len(uptake) != 1:
            _raise("estimation.armstats.arm_stats.combine_uptake_family")
        roles = {a.x_role for a in arms}
        if len(roles) != 1:
            _raise(
                "estimation.armstats.arm_stats.combine_partitions_declaring",
                roles=sorted(str(r) for r in roles),
            )

        views = [a.moments for a in arms]
        if head.ref_x is not None and head.x_role is None:
            # A role-less x slot (a clustered row predating the declaration) does
            # not survive pooling: the pooled row could not say what it carries.
            views = [view.without(X_UNDECLARED) for view in views]
        pooled = CenteredMoments.combine(views)

        winsorized = any(getattr(a, "winsor_n", None) is not None for a in arms)
        if winsorized:
            for a in arms[1:]:
                for field in (
                    "winsor_lower_percentile",
                    "winsor_upper_percentile",
                    "winsor_lower_bound",
                    "winsor_upper_bound",
                ):
                    if getattr(a, field) != getattr(head, field):
                        _raise(
                            "estimation.armstats.arm_stats.combine_inconsistent_winsorization",
                            field=field,
                        )
            winsor_lower_percentile = head.winsor_lower_percentile
            winsor_upper_percentile = head.winsor_upper_percentile
            winsor_lower_bound = head.winsor_lower_bound
            winsor_upper_bound = head.winsor_upper_bound
            winsor_n = sum(a.winsor_n or 0 for a in arms)
            winsor_n_lower = sum(a.winsor_n_lower or 0 for a in arms)
            winsor_n_upper = sum(a.winsor_n_upper or 0 for a in arms)
        else:
            (
                winsor_lower_percentile,
                winsor_upper_percentile,
                winsor_lower_bound,
                winsor_upper_bound,
                winsor_n,
                winsor_n_lower,
                winsor_n_upper,
            ) = (None,) * 7

        return cls(
            study_id=head.study_id,
            metric=head.metric,
            group_id=group_id,
            n=pooled.n,
            **slot_fields(pooled),
            successes=pooled.successes,
            winsor_lower_percentile=winsor_lower_percentile,
            winsor_upper_percentile=winsor_upper_percentile,
            winsor_lower_bound=winsor_lower_bound,
            winsor_upper_bound=winsor_upper_bound,
            winsor_n=winsor_n,
            winsor_n_lower=winsor_n_lower,
            winsor_n_upper=winsor_n_upper,
        )

    # First moments: exact recovery from reference + residual

    def mean_y(self) -> float:
        """Mean of *y*: ``ref_y + cy1 / n``, exact to the stored reference."""
        return self.moments.mean("y")

    def mean_x(self) -> float:
        """Mean of whatever the x slot carries (covariate, cluster size, uptake total)."""
        if self.ref_x is None or self.cx1 is None:
            _raise("estimation.armstats.arm_stats.needs_covariate_ref")
        x_var = self.x_variable
        assert x_var is not None
        return self.moments.mean(x_var)

    def mean_den(self) -> float:
        """Mean of the ratio denominator."""
        if self.ref_den is None or self.cden1 is None:
            _raise("estimation.armstats.arm_stats.mean_den_needs_ref_den")
        return self.moments.mean("den")

    def mean_d(self) -> float:
        """Uptake rate ``d-bar`` (binary *d*); refuses when the uptake fact is absent."""
        if self.sum_d is None:
            _raise("estimation.armstats.arm_stats.mean_d_needs_uptake_sum")
        return self.moments.mask_mean("d")

    def mean_yd(self) -> float:
        """Mean of ``y * d``: ``(cyd + ref_y * sum_d) / n``, a product, never a difference."""
        self._require_uptake()
        return self.moments.masked_mean("y", "d")

    def mean_y_untaken(self) -> float:
        """Mean of ``y * (1 - d)``, expanded so two large means are never subtracted."""
        self._require_uptake()
        return self.moments.unmasked_mean("y", "d")

    # Variance reductions: every / (n-1) lives on CenteredMoments

    def _label(self) -> str:
        return f"metric={self.metric!r} group={self.group_id!r}"

    def var_y(self) -> float:
        """ddof=1 variance of *y*, exact-grade at every scale (centered by the producer)."""
        self._require_two_units("ddof=1 variance")
        return self.moments.var("y", what=f"var_y for {self._label()}")

    def var_x(self) -> float:
        """ddof=1 variance of the x slot."""
        if self.cx2 is None or self.cx1 is None:
            _raise("estimation.armstats.arm_stats.needs_covariate_cx2")
        self._require_two_units("ddof=1 variance")
        x_var = self.x_variable
        assert x_var is not None
        return self.moments.var(x_var, what=f"var_x for {self._label()}")

    def var_den(self) -> float:
        """ddof=1 variance of the ratio denominator."""
        if self.cden2 is None or self.cden1 is None:
            _raise("estimation.armstats.arm_stats.var_den_needs_cden2")
        self._require_two_units("ddof=1 variance")
        return self.moments.var("den", what=f"var_den for {self._label()}")

    def cov_yx(self) -> float:
        """ddof=1 covariance of *y* and the x slot."""
        if self.cxy is None or self.cx1 is None:
            _raise("estimation.armstats.arm_stats.needs_covariate_cxy")
        self._require_two_units("ddof=1 covariance")
        x_var = self.x_variable
        assert x_var is not None
        return self.moments.cov(x_var, "y")

    def cov_yden(self) -> float:
        """ddof=1 covariance of *y* and the ratio denominator."""
        if self.cyden is None or self.cden1 is None:
            _raise("estimation.armstats.arm_stats.cov_yden_needs_cyden")
        self._require_two_units("ddof=1 covariance")
        return self.moments.cov("y", "den")

    def cov_xden(self) -> float:
        """ddof=1 covariance of the x slot and the ratio denominator (the
        cluster-robust LATE collapse reads cluster uptake total vs cluster size here)."""
        if self.cxden is None or self.cx1 is None or self.cden1 is None:
            _raise("estimation.armstats.arm_stats.needs_denominator_cross")
        self._require_two_units("ddof=1 covariance")
        x_var = self.x_variable
        assert x_var is not None
        return self.moments.cov(x_var, "den")

    def _require_uptake(self) -> None:
        if self.sum_d is None or self.cyd is None or self.cy2d is None:
            _raise("estimation.armstats.arm_stats.needs_uptake_cyd")

    def _require_two_units(self, quantity: str) -> None:
        """A 1-unit arm has no ddof=1 spread; name the arm and the metric
        so a downstream complaint doesn't leave the caller hunting for the
        stray arm.
        """
        if self.n < 2:
            _raise(
                "estimation.armstats.arm_stats.least_compute_metric",
                quantity=quantity,
                group_id=self.group_id,
                metric=self.metric,
                n=self.n,
            )

    def var_d(self) -> float:
        """ddof=1 variance of binary *d*: ``sum_d * (n - sum_d) / n / (n - 1)``."""
        if self.sum_d is None:
            _raise("estimation.armstats.arm_stats.mean_d_needs_uptake_sum")
        self._require_two_units("ddof=1 variance")
        return self.moments.mask_var("d")

    def cov_yd(self) -> float:
        """ddof=1 covariance of *y* and binary *d*: ``cyd - cy1*sum_d/n``, nothing large formed."""
        if self.sum_d is None or self.cyd is None:
            _raise("estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd")
        self._require_two_units("ddof=1 covariance")
        return self.moments.cov_mask("y", "d")

    def cov_xd(self) -> float:
        """ddof=1 covariance of the x slot and binary *d* (the CUPED-adjusted LATE cross term)."""
        if self.cx1 is None or self.sum_d is None or self.cxd is None:
            _raise("estimation.armstats.arm_stats.needs_covariate_uptake")
        self._require_two_units("ddof=1 covariance")
        x_var = self.x_variable
        assert x_var is not None
        return self.moments.cov_mask(x_var, "d")

    def var_yd(self) -> float:
        """ddof=1 variance of the product ``y*d`` (see CenteredMoments.masked_product_var)."""
        self._require_uptake()
        self._require_two_units("ddof=1 variance")
        return self.moments.masked_product_var("y", "d", what=f"var_yd for {self._label()}")

    def cov_y_yd(self) -> float:
        """ddof=1 covariance of *y* with ``y*d`` (see CenteredMoments.cov_masked_product)."""
        self._require_uptake()
        self._require_two_units("ddof=1 covariance")
        return self.moments.cov_masked_product("y", "d")

    def to_summary(self) -> SummaryStats:
        """Reduce to (n, mean, ddof=1 var)."""
        return SummaryStats(n=self.n, mean=self.mean_y(), var=self.var_y())


# prose: allow-long Bernoulli second-moment tolerance derives from the producer's summation error
#: `arm.cy2` chains window `AVG` and residual `SUM` passes, but `variance_slack` bounds one
#: accumulation at its typical `sqrt(n)` drift. A sum of `n` nonnegative terms errs by at most
#: `gamma_(n-1)` of itself in any summation order, where `gamma_k = k u / (1 - k u)` (`u = 2**-53`)
#: is the error of `k` compounded roundings (the denominator is their second-order growth, which
#: `k u` alone omits: `(k u)**2`, 1.2e-14 of itself at 1e9 units against `8 u` of headroom, 8.9e-16).
#: About `7 u` more rounds each term and the expected `successes * (n - successes) / n`, so
#: `gamma_(n+8)` bounds the error. Producers approach it, adding millions of equal tiny residuals
#: onto a growing total that rounds the same way every time. On the real DuckDB producer path
#: (`scripts/measure_binomial_ceiling.py recovery`) the error reached 2442x `variance_slack` at
#: 1e9 units and stayed within a quarter of the bound at every size (at most 0.24 of it, at 4e6
#: units). `1024 * variance_slack` keeps >4x headroom up to 4e6 units and exceeds the bound
#: there; above, the bound is the tolerance. Corrupt or non-binary data misses either by orders
#: of magnitude.
_BERNOULLI_CONSISTENCY_SLACK = 1024.0
_UNIT_ROUNDOFF = 2.0**-53

#: Absolute floor on the consistency check of a stored residual first moment.
_FIRST_MOMENT_ABSOLUTE_TOLERANCE = Fraction(1e-6)


def _float_or_inf(value: Fraction) -> float:
    """``float(value)``, an infinity of its sign where the value exceeds float64."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _rounding_bound(units: int) -> float:
    """``gamma_k = k u / (1 - k u)``, the relative error of *units* compounded roundings;
    ``math.inf`` once ``k u`` reaches one, where no relative bound holds."""
    product = units * _UNIT_ROUNDOFF
    return product / (1.0 - product) if product < 1.0 else math.inf


def binary_counts(arm: ArmStats, metric_type: str) -> tuple[int, int]:
    """Read exact ``(successes, trials)`` for a declared, unadjusted binary outcome.

    Counts travel as integers alongside the centered moments. Rounded residuals do not
    identify a success count at large scales; an arm missing that information must be
    rebuilt from its original data. Metric provenance and independent-unit eligibility
    remain separate requirements, and moment consistency is a corruption check, not proof
    that an undeclared outcome is Bernoulli.
    """
    if metric_type not in ("conversion", "retention"):
        _raise("estimation.binomial.binary_provenance_required", metric_type=metric_type)
    if arm.ref_x is not None or arm.ref_den is not None or arm.sum_d is not None:
        _raise(
            "estimation.binomial.independent_units_required",
            metric=arm.metric,
            group_id=arm.group_id,
        )
    successes = arm.successes
    if successes is None:
        _raise(
            "estimation.binomial.exact_counts_required",
            metric=arm.metric,
            group_id=arm.group_id,
        )
    if isinstance(successes, bool) or not isinstance(successes, numbers.Integral):
        _raise(
            "estimation.binomial.reconstructed_counts_not_binary",
            metric=arm.metric,
            group_id=arm.group_id,
            sum_y=successes,
            n=arm.n,
        )
    successes = int(successes)
    reference = Fraction(arm.ref_y)
    sum_y = arm.n * reference + Fraction(arm.cy1)
    # Each producer rounds its per-unit subtraction and then sums the residuals.
    # Counts are authoritative; this bounds their disagreement with that noisy sum.
    spread = successes * abs(1 - reference) + (arm.n - successes) * abs(reference)
    rounding = _rounding_bound(arm.n + 8)
    consistent = not math.isfinite(rounding) or abs(sum_y - successes) <= max(
        _FIRST_MOMENT_ABSOLUTE_TOLERANCE,
        Fraction(math.nextafter(rounding, math.inf)) * spread,
    )
    if not consistent or not 0 <= successes <= arm.n:
        _raise(
            "estimation.binomial.reconstructed_counts_not_binary",
            metric=arm.metric,
            group_id=arm.group_id,
            sum_y=_float_or_inf(sum_y),
            n=arm.n,
        )
    # A mean alone cannot distinguish 0/1 draws from, e.g., constant 0.5 outcomes. Bernoulli
    # data's centered sum of squares is exactly `successes * (n - successes) / n`; a gap beyond
    # the aggregation slack means corrupt or mismatched input.
    expected_cy2 = successes * (arm.n - successes) / arm.n
    magnitude = max(abs(arm.cy2), abs(expected_cy2), 1.0)
    tolerance = max(
        _BERNOULLI_CONSISTENCY_SLACK * variance_slack(magnitude, arm.n),
        _rounding_bound(arm.n + 8) * magnitude,
    )
    if abs(arm.cy2 - expected_cy2) > tolerance:
        _raise(
            "estimation.binomial.inconsistent_bernoulli_variance",
            metric=arm.metric,
            group_id=arm.group_id,
            cy2=arm.cy2,
            expected_cy2=expected_cy2,
            n=arm.n,
            tolerance=tolerance,
        )
    return successes, arm.n


def canonical_bernoulli_arm(arm: ArmStats, successes: int) -> ArmStats:
    """``arm`` with its y family replaced by the exact moments of ``successes`` ones among its
    ``arm.n`` units, every other field kept.

    A declared binary arm is determined by its integer counts. Forming the y family from
    those counts removes producer-dependent rounding in the stored moments, so runtime
    and planning decide the same count pair identically.

    The y family is ``ArmStats``' own centering, about the stored reference ``ref_y`` (the
    correctly rounded rate): ``cy1 = successes - n * ref_y`` is exact, and ``cy2`` is the exact
    centered sum of squares about the true mean, a rational of the integer counts rounded once,
    plus the squared offset of ``ref_y`` from it, ``cy1**2 / n``. Nothing is cancelled between
    raw sums, so the noise floor that polices a raw sum of squares does not apply: an arm with
    many units and few failures keeps the variance of its failures, which a raw-sum centering
    clamps to zero once ``n - successes`` falls below about ``8 * eps * n``. Counts outside
    ``[0, n]`` are not a count of this arm's units and are refused as non-binary."""
    n = arm.n
    if (
        isinstance(successes, bool)
        or not isinstance(successes, numbers.Integral)
        or not 0 <= successes <= n
    ):
        _raise(
            "estimation.binomial.reconstructed_counts_not_binary",
            metric=arm.metric,
            group_id=arm.group_id,
            sum_y=successes,
            n=n,
        )
    ref_y = successes / n
    cy1 = float(Fraction(successes) - n * Fraction(ref_y))
    cy2 = float(Fraction(successes * (n - successes), n)) + scaled_cross_over_n(cy1, cy1, n)
    return arm.model_copy(
        update={"ref_y": ref_y, "cy1": cy1, "cy2": cy2, "successes": int(successes)}
    )


def centered_row_from_raw_sums(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one format-1 wire row (raw additive sums) to format 2.

    Non-moment keys pass through untouched; :data:`RAW_SUM_FIELDS` columns
    are replaced by :data:`CENTERED_FIELDS`.
    """
    unsupported = sorted({"sum_w", "sum_w2"}.intersection(row))
    if unsupported:
        _raise("estimation.armstats.centered_raw_row", unsupported=unsupported)
    out = {k: v for k, v in row.items() if k not in RAW_SUM_FIELDS}
    arm = ArmStats.from_raw_sums(
        study_id=str(row.get("experiment_id", "")),
        metric=str(row.get("metric", "")),
        group_id=str(row.get("group_id", "")),
        n=int(row["n"]),  # type: ignore[call-overload]
        sum_y=float(row["sum_y"]),  # type: ignore[arg-type]
        sum_y2=float(row["sum_y2"]),  # type: ignore[arg-type]
        successes=row.get("successes"),
        **{f: _opt_float(row.get(f)) for f in RAW_SUM_FIELDS[2:]},  # type: ignore[arg-type]
    )
    out.update({f: getattr(arm, f) for f in CENTERED_FIELDS})
    return out


def _opt_float(v: Any) -> float | None:
    """None/NaN -> None; anything else -> float."""
    if v is None or v != v:
        return None
    return float(v)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.armstats.centered_sum_squares": "{what}: centered sum of squares is {value:.6g}, negative beyond floating-point rounding (~{variance_slack:.3g}). Raw float64 sums lose the variance signal once spread/|mean| (the coefficient of variation) drops toward sqrt(eps) ~ 1e-8 -- but a deficit this large means the moment sums are inconsistent (mis-aggregated or corrupted upstream), not merely cancelled.",
        "estimation.armstats.cross_moment_materially": "{what}: cross moment {cross:.6g} is materially nonzero against a zero variance (var_a={var_a:.6g}, var_b={var_b:.6g}), which no real sample can produce -- the exact Cauchy-Schwarz bound is 0 and rounding at this scale is ~{floor:.3g}; these moments are corrupt, mis-aggregated, or drawn from mismatched partitions.",
        "estimation.armstats.cross_moment_violates": RefusalSpec(
            "estimation.armstats.cross_moment_violates",
            InvalidRequestError,
            lambda *, what, cross, var_a, var_b, bound, excess, variance_slack: (
                f"{what}: cross moment {cross:.6g} violates the Cauchy-Schwarz bound against "
                f"variances {var_a:.6g} and {var_b:.6g} (|cross|={abs(cross):.6g} exceeds "
                f"sqrt(var_a)*sqrt(var_b)={bound:.6g} by {excess:.6g}, beyond floating-point "
                f"rounding ~{variance_slack:.3g}) -- these moments cannot come from a real "
                "sample; they are corrupt, mis-aggregated, or drawn from mismatched partitions."
            ),
        ),
        "estimation.armstats.summary_stats.mean_finite": "mean must be finite, got {mean}",
        "estimation.armstats.summary_stats.var_finite": "var must be finite, got {var}",
        "estimation.armstats.arm_stats.obsolete_armstats_fields": "obsolete ArmStats fields are not supported: {obsolete}; weighted estimands must use ScoreStats",
        "estimation.armstats.arm_stats.group_summary_moment": "group_summary moment {name!r} is non-finite ({value!r}) for metric={metric!r} group={group_id!r}; {name} must be finite",
        "estimation.armstats.arm_stats.partial_family_missing": "partial {label} family: has {present} but is missing {missing}",
        "estimation.armstats.arm_stats.cross_field_finite": "{name} must be finite, got {value!r}",
        "estimation.armstats.arm_stats.x_role_one": "x_role must be one of 'covariate', 'cluster_size', or 'uptake_total'; got {x_role!r}",
        "estimation.armstats.arm_stats.x_role_declared": "x_role must be declared when the x family is present",
        "estimation.armstats.arm_stats.x_role_family": "x_role requires the x family to be present",
        "estimation.armstats.arm_stats.cxden_complete_denominator": "cxden requires complete x and denominator prerequisite families",
        "estimation.armstats.arm_stats.cxd_complete_uptake": "cxd requires complete x and uptake prerequisite families",
        "estimation.armstats.arm_stats.sum_d_outside": "sum_d={sum_d!r} is outside [0, n={n}] for metric={metric!r} group={group_id!r} -- an uptake count cannot be negative or exceed the arm's own unit count",
        "estimation.armstats.arm_stats.sum_d_zero_but_masked_nonzero": "sum_d=0 for metric={metric!r} group={group_id!r} but {nonzero} nonzero -- no unit took up, so every uptake-masked moment must be exactly zero; these moments were not produced from the same mask",
        "estimation.armstats.arm_stats.winsorization_metadata_winsor": "winsorization metadata requires winsor_n and both cap counts",
        "estimation.armstats.arm_stats.winsorization_cap_counts": "winsorization cap counts cannot exceed winsor_n",
        "estimation.armstats.arm_stats.winsorization_metadata_finite": "winsorization metadata {name} must be finite",
        "estimation.armstats.arm_stats.winsorization_lower_percentile": "winsorization lower percentile requires a resolved bound",
        "estimation.armstats.arm_stats.winsorization_upper_percentile": "winsorization upper percentile requires a resolved bound",
        "estimation.armstats.arm_stats.winsorization_resolved_lower": "winsorization resolved lower bound must be below upper bound",
        "estimation.armstats.arm_stats.combine_needs_least": "combine() needs at least one partition",
        "estimation.armstats.arm_stats.combine_needs_one": "combine() needs one study and one metric; got {head_study_id!r}/{head_metric!r} and {a_study_id!r}/{a_metric!r}",
        "estimation.armstats.arm_stats.combine_partitions_from": "combine() got partitions from different arms ({labels}); pass group_id= to label the combined record explicitly.",
        "estimation.armstats.arm_stats.combine_family_some": "combine() got the {family} family on some partitions and not others ({probe} mixed None/non-None); partitions of one population must agree on which families exist.",
        "estimation.armstats.arm_stats.combine_uptake_family": "combine() got the uptake family on some partitions and not others (sum_d mixed None/non-None); partitions of one population must agree on which families exist.",
        "estimation.armstats.arm_stats.combine_partitions_declaring": "combine() got partitions declaring different x-family roles ({roles}); partitions of one population must agree on what the x family carries (covariate / cluster size / uptake total).",
        "estimation.armstats.arm_stats.combine_inconsistent_winsorization": "combine() got inconsistent winsorization metadata in {field}",
        "estimation.armstats.arm_stats.needs_covariate_ref": "needs covariate: ref_x/cx1 is None (CUPED not materialised)",
        "estimation.armstats.arm_stats.mean_den_needs_ref_den": "ratio-family moments require ref_den/cden1 (not None)",
        "estimation.armstats.arm_stats.mean_d_needs_uptake_sum": "needs uptake: sum_d is None (uptake fact not materialised)",
        "estimation.armstats.arm_stats.centered_sum_squares": "{what}: centered sum of squares is {c2:.6g}, but it is a sum of squares and cannot be negative -- these moments are corrupt or were not produced by a moments builder.",
        "estimation.armstats.arm_stats.needs_covariate_cx2": "needs covariate: cx2/cx1 is None (CUPED not materialised)",
        "estimation.armstats.arm_stats.var_den_needs_cden2": "ratio-family moments require cden2/cden1 (not None)",
        "estimation.armstats.arm_stats.needs_covariate_cxy": "needs covariate: cxy/cx1 is None (CUPED not materialised)",
        "estimation.armstats.arm_stats.cov_yden_needs_cyden": "ratio-family moments require cyden/cden1 (not None)",
        "estimation.armstats.arm_stats.needs_denominator_cross": "needs x-denominator cross moment: cxden is None",
        "estimation.armstats.arm_stats.needs_uptake_cyd": "needs uptake: cyd/cy2d is None (uptake fact not materialised)",
        "estimation.armstats.arm_stats.least_compute_metric": "n must be at least 2 to compute {quantity} (metric {metric!r}, arm {group_id!r} has n={n})",
        "estimation.armstats.arm_stats.cov_yd_needs_uptake_sum_cyd": "needs uptake: sum_d/cyd is None (uptake fact not materialised)",
        "estimation.armstats.arm_stats.needs_covariate_uptake": "needs covariate x uptake: cxd is None. Both a CUPED covariate and an uptake fact must be materialised on the same arm: from a definitions path, set n_pre_periods > 0 on the experiment; from from_unit_summary, pass MetricSpec(name=..., covariate='<pre-period column>') alongside uptake='<column>'.",
        "estimation.armstats.centered_raw_row": "centered/raw row carries unsupported weighted ArmStats fields: {unsupported}; weighted estimands must use ScoreStats.",
        "estimation.armstats.score_stats.cluster_variance": "cluster_variance and n_clusters must be set together -- a clustered second moment without its K has no coherent SE.",
        "estimation.armstats.score_stats.n_clusters_clustered": "n_clusters must be >= 1, got {n_clusters} -- a clustered SE with no clusters has no coherent reference.",
        "estimation.armstats.score_stats.score_scale": "{field} must be finite and positive, got {value!r}",
        "estimation.armstats.score_stats.cluster_score_scale": "cluster_score_scale requires cluster_variance and n_clusters -- a cluster scale without clustered moments has no meaning.",
        "estimation.armstats.score_stats.se_needs_positive": "se() needs a positive normalizer, got {normalizer}",
        "estimation.binomial.binary_provenance_required": "binary_counts requires a declared conversion/retention metric type (a structural 0/1-per-unit fact), got metric_type={metric_type!r}: matching moments alone never proves a metric is genuinely Bernoulli",
        "estimation.binomial.independent_units_required": "binary_counts(metric={metric!r}, group_id={group_id!r}): the arm carries a CUPED covariate, ratio denominator, or uptake-mask family -- clustered, weighted, adjusted, or otherwise non-unit-grain observations are not independent Bernoulli units this method admits",
        "estimation.binomial.exact_counts_required": RefusalSpec(
            "estimation.binomial.exact_counts_required",
            CapabilityError,
            template="binary_counts(metric={metric!r}, group_id={group_id!r}) requires an exact integer successes count; rebuild from original unit data or re-export current moments, because floating centered moments cannot recover lost count information",
        ),
        "estimation.binomial.reconstructed_counts_not_binary": RefusalSpec(
            "estimation.binomial.reconstructed_counts_not_binary",
            BinomialDataError,
            template="binary_counts(metric={metric!r}, group_id={group_id!r}): success total {sum_y!r} over n={n!r} is not a valid integer count consistent with the binary moments",
        ),
        "estimation.binomial.inconsistent_bernoulli_variance": RefusalSpec(
            "estimation.binomial.inconsistent_bernoulli_variance",
            BinomialDataError,
            template="binary_counts(metric={metric!r}, group_id={group_id!r}): centered sum of squares cy2={cy2!r} does not match the Bernoulli-consistent value {expected_cy2!r} for n={n!r} within tolerance {tolerance!r} -- this arm's y values are not genuinely independent 0/1 draws (matching moments alone do not prove Bernoulli provenance; this is a corruption guard on top of the declared metric type)",
        ),
        "estimation.armstats.independent_mean_contract": RefusalSpec(
            "estimation.armstats.independent_mean_contract",
            InvalidRequestError,
            lambda *, reason: reason,
        ),
    },
)
_raise = raiser(_REFUSALS)


class IndependentMeanComponent(CodedModel, BaseModel):
    """One unweighted independent iid mean, with centered squared deviations."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    group_id: str
    n: int = Field(gt=1, strict=True)
    mean: float
    centered_sum_squares: float = Field(ge=0)
    coefficient: float

    @classmethod
    def from_values(cls, group_id: str, values: Sequence[float], *, coefficient: float):
        if len(values) < 2 or any(not math.isfinite(y) for y in values):
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Independent means require at least two finite observations per component.",
            )
        try:
            center = values[0]
            offsets = [float(y) - center for y in values]
            shift = math.fsum(d / len(values) for d in offsets)
            mean = center + shift
            ss = math.fsum((d - shift) ** 2 for d in offsets)
        except (OverflowError, ValueError):
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Centered mean component exceeds finite numeric range.",
            )
        if not math.isfinite(mean) or not math.isfinite(ss):
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Centered mean component exceeds finite numeric range.",
            )
        return cls(
            group_id=group_id,
            n=len(values),
            mean=mean,
            centered_sum_squares=ss,
            coefficient=coefficient,
        )

    @classmethod
    def from_arm_stats(cls, arm: ArmStats, *, coefficient: float):
        if (
            arm.ref_x is not None
            or arm.ref_den is not None
            or arm.sum_d is not None
            or arm.winsor_upper_percentile is not None
            or arm.winsor_lower_percentile is not None
        ):
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Independent mean components cannot contain adjusted covariance state.",
            )
        summary = arm.to_summary()
        return cls(
            group_id=arm.group_id,
            n=summary.n,
            mean=summary.mean,
            centered_sum_squares=summary.var * (summary.n - 1),
            coefficient=coefficient,
        )

    @property
    def variance(self) -> float:
        value = (
            Fraction(self.coefficient) ** 2
            * Fraction(self.centered_sum_squares)
            / (self.n * (self.n - 1))
        )
        try:
            return float(value)
        except OverflowError:
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Component variance exceeds finite numeric range.",
            )


class IndependentMeanReference(CodedModel, BaseModel):
    """Welch components and the requested inference contract.

    Alpha is the caller's tail budget. Directional central displays use
    doubled alpha; directional-FCR displays spend that display budget on
    the single finite endpoint. Adjusted scores use a separate reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    kind: Literal["independent-means-welch-v2"] = "independent-means-welch-v2"
    interval: Literal["central", "directional-fcr"] = "central"
    alpha: float = Field(default=0.05, gt=0, lt=1)
    alternative: Alternative = "two-sided"
    components: tuple[IndependentMeanComponent, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _migrate_v1(cls, data):
        if isinstance(data, dict) and data.get("kind", "independent-means-welch-v1") == (
            "independent-means-welch-v1"
        ):
            return {**data, "kind": "independent-means-welch-v2"}
        return data

    @model_validator(mode="after")
    def _components(self):
        if len({c.group_id for c in self.components}) != len(self.components):
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Independent mean component identities must be distinct.",
            )
        if not math.isfinite(self.point) or not math.isfinite(self.variance) or self.variance <= 0:
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Independent mean contrast requires positive finite variance.",
            )
        return self

    @property
    def point(self) -> float:
        try:
            return float(
                sum(
                    (Fraction(c.coefficient) * Fraction(c.mean) for c in self.components),
                    Fraction(),
                )
            )
        except OverflowError:
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Independent mean contrast exceeds finite numeric range.",
            )

    @property
    def variance(self) -> float:
        try:
            return math.fsum(c.variance for c in self.components)
        except OverflowError:
            _raise(
                "estimation.armstats.independent_mean_contract",
                reason="Contrast variance exceeds finite numeric range.",
            )

    def se(self) -> float:
        return math.sqrt(self.variance)

    @property
    def df(self) -> float:
        scale = max(c.variance for c in self.components)
        parts = tuple(c.variance / scale for c in self.components)
        return math.fsum(parts) ** 2 / math.fsum(
            v * v / (c.n - 1) for v, c in zip(parts, self.components, strict=True)
        )


def _restore_score_scale(root: float, scale: float, normalizer: float) -> float:
    """Restore a moment scale after square-rooting without forming scale²."""
    if scale == 1.0:
        return root / normalizer
    return float(Fraction(root) * Fraction(scale) / Fraction(normalizer))


class ScoreStats(CodedModel, BaseModel):
    """One contrast's collapsed influence scores: the second typed seam.

    Where ``ArmStats`` holds centered per-arm moments reduced with
    ``/(n-1)`` variance formulas, ``ScoreStats`` holds pre-summed
    estimating-equation ("influence function") values for one contrast
    (one treatment arm vs control), evaluated at the fitted solution.
    ``sum_psi`` is exactly (up to float error) zero under the Hajek
    normalization IPTW uses, since both arm-level score components are
    individually mean-zero; the field is carried through for a future
    Horvitz-Thompson registration whose ``sum_psi`` would be genuinely
    nonzero, since ``se()`` centers on it.

    Scores are additive across any partition of the contributing units,
    so per-fold or per-segment ``(sum_psi, sum_psi2)`` pairs from one fit
    reproduce the pooled moments exactly.  The stored moments are in
    units of ``score_scale`` (that is, moments of ``psi / score_scale``),
    which prevents forming an otherwise overflowing square.  Additive
    partitions must use a common scale; if independently produced
    partitions use different scales, rebase their moments before adding
    them.  No merge operation is implied by this metadata.

    ``se()`` divides by ``n`` (IPTW's normalizer) unless ``sum_d_tilde2``
    is supplied, in which case that residual-variation sum is the
    normalizer instead. ``cluster_variance``/``n_clusters`` are the
    cluster-robust seam: when a randomization-grain cluster is declared,
    the caller (``_adjust.overlap._cluster_reduction``) resolves the
    final covariance numerator, including any caller-owned scaling. The
    adjusted superpopulation path sums complete centered cluster scores,
    preserving both cross-arm covariance and heterogeneous target variation.
    ``cluster_variance`` is in units of ``cluster_score_scale`` squared;
    ``se()`` restores that scale without forming its square.  A missing
    ``cluster_score_scale`` means ``score_scale``. ``se()`` takes its square
    root and divides by the estimator's normalizer; it never adds a second
    correction. Reference metadata travels separately.
    ``n_clusters`` is the total count of independent clusters.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    contrast: str  # the treatment group_id being compared to the design's control
    n: int
    sum_psi: float
    sum_psi2: float
    score_scale: float = 1.0
    sum_d_tilde2: float | None = None
    cluster_variance: float | None = None
    n_clusters: int | None = None
    cluster_score_scale: float | None = None

    @model_validator(mode="after")
    def _cluster_fields_travel_together(self) -> ScoreStats:
        if not math.isfinite(self.score_scale) or self.score_scale <= 0.0:
            _raise(
                "estimation.armstats.score_stats.score_scale",
                field="score_scale",
                value=self.score_scale,
            )
        if self.cluster_score_scale is not None and (
            not math.isfinite(self.cluster_score_scale) or self.cluster_score_scale <= 0.0
        ):
            _raise(
                "estimation.armstats.score_stats.score_scale",
                field="cluster_score_scale",
                value=self.cluster_score_scale,
            )
        if (self.cluster_variance is None) != (self.n_clusters is None):
            _raise("estimation.armstats.score_stats.cluster_variance")
        if self.cluster_score_scale is not None and self.cluster_variance is None:
            _raise("estimation.armstats.score_stats.cluster_score_scale")
        if self.n_clusters is not None and self.n_clusters < 1:
            _raise(
                "estimation.armstats.score_stats.n_clusters_clustered", n_clusters=self.n_clusters
            )
        return self

    def se(self) -> float:
        """Standard error: ``sqrt(centered sum of squared scores) /
        normalizer``.

        The second moment is centered via the exact-Fraction helper
        (:func:`centered_sq_sum`), not a raw ``sum_psi2 - sum_psi**2/n``
        subtraction: for scores offset far from zero, that raw
        subtraction cancels catastrophically and can silently report a
        standard error of exactly 0.0 for a genuinely nonzero true SE.
        A no-op for the mean-zero scores every current registration
        produces, and correct rather than inflated by ``n * (mean
        psi)**2`` for any registration whose scores are not mean-zero.

        ``normalizer`` is ``n`` unless ``sum_d_tilde2`` is set, in which
        case it is the normalizer instead (DML's sandwich denominator).
        With ``cluster_variance`` set, that already-resolved number
        replaces the unit-level moment verbatim (no further multiplier):
        the caller resolved it jointly with the matching reference df.
        """
        normalizer = self.n if self.sum_d_tilde2 is None else self.sum_d_tilde2
        if normalizer <= 0:
            _raise("estimation.armstats.score_stats.se_needs_positive", normalizer=normalizer)
        if self.cluster_variance is not None:
            cluster_scale = (
                self.score_scale if self.cluster_score_scale is None else self.cluster_score_scale
            )
            return _restore_score_scale(
                math.sqrt(max(self.cluster_variance, 0.0)),
                cluster_scale,
                normalizer,
            )
        centered = centered_sq_sum(
            self.sum_psi2,
            self.sum_psi,
            self.n,
            what=f"ScoreStats.se for metric={self.metric!r} contrast={self.contrast!r}",
            clamp_positive=True,
        )
        return _restore_score_scale(math.sqrt(centered), self.score_scale, normalizer)


ARM_STATS_CROSS_FIELD_FINITE = _REFUSALS["estimation.armstats.arm_stats.cross_field_finite"]
