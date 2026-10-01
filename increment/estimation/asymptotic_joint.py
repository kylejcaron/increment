"""Delta-method AsympCS inversion over retained joint per-arm moments.

One route serves three laws. Each arm retains an exact ``GaussianState`` of
dimension ``d`` (count, mean vector, full ``d x d`` centered scatter) and the
contrast is the ratio ``r = f(mu_t) / f(mu_c)`` of one per-arm functional:

* ``adjusted_mean`` over ``(Y, X)``: ``f = Ybar - theta * (Xbar - Xbar_pooled)``
  with ``theta`` the inverse-n weighted within-arm ``Cov(Y, X) / Var(X)`` that
  ``estimation.cuped`` fits at a fixed horizon, read here from the retained
  cross moments at every look.
* ``ratio_mean`` over ``(N, D)``: ``f = Nbar / Dbar``.
* ``adjusted_ratio_mean`` over ``(N, D, X)``: ``f = N' / D'`` with each
  component adjusted against X by its own coefficient.

Construction
------------
The scalar route (``asymptotic_mean``) inverts ``(m_t - r m_c)^2 <= K_N V(r)``
in ratio coordinates with ``V(r) = S_t/n_t^2 + r^2 S_c/n_c^2``. Here the
per-arm functionals are linearised at the observed means, so the contrast
``D(r) = f_t - r f_c`` has, with respect to the two retained mean vectors,
gradients ``G_a(r)`` that are affine in ``r``, and

    V(r) = sum_a G_a(r)' S_a G_a(r) / n_a^2

is the plug-in delta-method variance of ``D(r)``. ``D(r)^2 - K_N V(r)`` is
then a quadratic in ``r`` and the same outward quadratic solver, count-clock
boundary ``K_N`` and set geometry as the scalar route apply unchanged. The
pooled covariate anchor is shared by both arms, so for the adjusted laws
``G_a(r)`` carries the other arm's coefficient through the anchor; the
anchor-ignoring per-arm sum is exact only at ``r = 1`` and undercovers away
from it. For the ratio laws the gradient carries the ``-2 Cov(N, D)`` term
through the retained cross scatter.

What is exact and what is asymptotic
------------------------------------
The retained state, the coefficient, the anchor, the gradients, the boundary
``K_N`` and every quadratic coefficient are rational and computed exactly;
only the final display conversion rounds. The confidence sequence is
asymptotic in two respects: the count-clock boundary is the scalar route's
asymptotic construction (Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas,
arXiv:2103.06476), and the contrast is a first-order linearisation at the
observed means with nuisance plug-ins (``theta``, the observed denominator
means) treated as fixed. Under the declared contract (iid units, fixed
randomization probability, finite ``2 + delta`` moments of the retained
vector, positive limiting variance of the linearised contrast, positive
within-arm covariate variance for the adjusted laws, denominator mean bounded
away from zero for the ratio laws) the plug-in error is an order below the
boundary width: ``theta_hat - theta*`` and the covariate imbalance
``Xbar_t - Xbar_c`` are each ``O(sqrt(log log n / n))`` a.s. by the law of
the iterated logarithm, so their product moves the contrast by
``O(log log n / n)`` while the boundary shrinks like ``sqrt(log log n / n)``.
That is the heteroskedasticity-robust asymptotic construction for
regression-adjusted causal effects of Lindon, Ham, Tingley and Bojinov
(2022, arXiv:2210.08589), which covers a coefficient fitted from the
accumulated in-experiment outcomes, and the delta-method confidence
sequence for relative lift with regression adjustment of Schmit and Miller
(2022, https://svenschmit.com/assets/pdf/code_2022_ci.pdf).

Preparation and family evidence
-------------------------------
Everything above except ``K_N`` is a function of the stopped state alone, so
``prepare_joint`` computes it once and ``invert_joint`` reads it at any error
level. For a ratio law the linearisation is trusted only where each arm's
denominator is separated from zero at the look's boundary,
``Dbar^2 > K_N Var(Dbar)`` with the adjusted mean and variance from both
arms' gradients through the pooled covariate anchor. Set availability
depends on alpha. Family evidence must not: it is the
contrast's sign-gated mixture value capped by the denominators' strict
stability score (``asymptotic_mean._count_boundary_log_cap``), and that cap
clears ``-log(alpha)`` only where the guard passes at ``alpha``. The capped
value is dominated by the mixture value, not itself a martingale, and not the
dual of the displayed set.

Why the exact e-process route keeps refusing an in-experiment coefficient
-------------------------------------------------------------------------
``AlwaysValid`` builds an increment-level (super)martingale in the declared
filtration. A coefficient fitted from accumulated outcomes re-weights every
past increment with information that was unavailable when it was revealed,
so the supermartingale property is lost outright rather than degraded, and
no negligibility argument substitutes for it. The public exact Bernoulli
route also does not admit a predeclared adjustment: predictability of a
transform does not preserve the Bernoulli outcome law. Predeclared scalar
adjustments belong to the asymptotic scalar-mean route.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from increment._literals import ALTERNATIVE_VALUES, Alternative
from increment.errors import CapabilityError, RefusalSpec, refuse
from increment.estimation._sequential_likelihood import GaussianState
from increment.estimation.asymptotic_mean import (
    AsymptoticMeanSet,
    MeanSetComponent,
    _component_for_quadratic,
    _count_boundary_log_cap,
    boundary_alpha,
    count_boundary,
    count_boundary_log_e,
    one_sided_components,
)
from increment.semantics.sequential import ScalarMeanModel, retained_dimension

_Vector = tuple[Fraction, ...]
_ZERO = Fraction(0)
_ONE = Fraction(1)

_INVALID = RefusalSpec(
    "sequential.asymptotic_joint.invalid", CapabilityError, lambda *, reason: reason
)


@dataclass(frozen=True, slots=True)
class _Arm:
    """One arm's linearised functional and the affine-in-r contrast gradient.

    ``own`` is the gradient with the pooled anchor held fixed, the arm's
    limiting-variance direction; ``constant + r * slope`` is the gradient of
    ``f_t - r f_c`` with respect to this arm's mean vector, anchor included.
    """

    value: Fraction
    own: _Vector
    constant: _Vector
    slope: _Vector
    denominator: tuple[Fraction, _Vector, _Vector | None] | None


@dataclass(frozen=True, slots=True)
class JointLinearisation:
    control: _Arm
    treatment: _Arm


def _coefficient(control: GaussianState, treatment: GaussianState, y: int, x: int):
    """Inverse-n weighted within-arm ``Cov(Y, X) / Var(X)``; None when unidentified.

    Bessel-corrected retained moments, an arm whose covariate is constant
    excluded from both sums, exactly ``estimation.cuped._within_arm_theta``.
    """
    covariance = variance = _ZERO
    for arm in (control, treatment):
        if arm.n < 2 or not arm.scatter[x][x]:
            continue
        weight = Fraction(1, arm.n * (arm.n - 1))
        covariance += weight * arm.scatter[y][x]
        variance += weight * arm.scatter[x][x]
    return covariance / variance if variance else None


@dataclass(frozen=True, slots=True)
class _Functional:
    """One arm's value, outcome gradient, covariate-shift slope, and denominator
    triple (mean, own gradient, optional other-arm gradient)."""

    value: Fraction
    lead: _Vector
    phi: Fraction = _ZERO
    denominator: tuple[Fraction, _Vector, _Vector | None] | None = None


def _anchored(
    control: _Functional, treatment: _Functional, n_c: int, n_t: int, *, covariate: bool
) -> JointLinearisation:
    """Contrast gradients ``G_a(r)`` for functionals sharing one pooled anchor."""
    zeros = (_ZERO,) * len(control.lead)
    negated = tuple(-v for v in control.lead)
    if not covariate:
        return JointLinearisation(
            _Arm(control.value, control.lead, zeros, negated, control.denominator),
            _Arm(treatment.value, treatment.lead, treatment.lead, zeros, treatment.denominator),
        )
    n = n_c + n_t
    weight_t, weight_c = Fraction(n_c, n) * treatment.phi, Fraction(n_t, n) * control.phi

    def denominator(
        functional: _Functional, own_weight: Fraction
    ) -> tuple[Fraction, _Vector, _Vector | None] | None:
        if functional.denominator is None:
            return None
        mean, gradient, _ = functional.denominator
        if len(gradient) == 2:
            return mean, gradient, None
        # ``gradient[-1]`` is -theta_D.  The pooled anchor routes its
        # opposite slope to the other arm.
        own = (*gradient[:-1], gradient[-1] * own_weight)
        other = (*zeros, -gradient[-1] * own_weight)
        return mean, own, other

    return JointLinearisation(
        _Arm(
            control.value,
            (*control.lead, -control.phi),
            (*zeros, weight_t),
            (*negated, weight_c),
            denominator(control, Fraction(n_t, n)),
        ),
        _Arm(
            treatment.value,
            (*treatment.lead, -treatment.phi),
            (*treatment.lead, -weight_t),
            (*zeros, -weight_c),
            denominator(treatment, Fraction(n_c, n)),
        ),
    )


def _ratio(
    numerator: Fraction, denominator: Fraction, theta: tuple[Fraction, Fraction] | None
) -> _Functional:
    """``N/D`` linearised at observed component means; ``theta`` = (theta_N, theta_D)."""
    ratio = numerator / denominator
    by_n, by_d = 1 / denominator, -ratio / denominator
    if theta is None:
        return _Functional(ratio, (by_n, by_d), denominator=(denominator, (_ZERO, _ONE), None))
    theta_n, theta_d = theta
    return _Functional(
        ratio,
        (by_n, by_d),
        by_n * theta_n + by_d * theta_d,
        (denominator, (_ZERO, _ONE, -theta_d), None),
    )


def linearise(
    control: GaussianState, treatment: GaussianState, declaration: ScalarMeanModel
) -> tuple[JointLinearisation | None, str | None]:
    """Linearise the declared law at the observed means, or name why not."""
    law = declaration.law
    n_c, n_t = control.n, treatment.n
    if law == "ratio_mean":
        if not control.mean[1] or not treatment.mean[1]:
            return None, "zero_denominator_mean"
        parts = [_ratio(arm.mean[0], arm.mean[1], None) for arm in (control, treatment)]
        return _anchored(parts[0], parts[1], n_c, n_t, covariate=False), None
    x = retained_dimension(law) - 1
    anchor = (n_c * control.mean[x] + n_t * treatment.mean[x]) / (n_c + n_t)
    if law == "adjusted_mean":
        theta = _coefficient(control, treatment, 0, x)
        if theta is None:
            return None, "zero_covariate_variance"
        parts = [
            _Functional(arm.mean[0] - theta * (arm.mean[x] - anchor), (_ONE,), theta)
            for arm in (control, treatment)
        ]
        return _anchored(parts[0], parts[1], n_c, n_t, covariate=True), None
    theta_n = _coefficient(control, treatment, 0, x)
    theta_d = _coefficient(control, treatment, 1, x)
    if theta_n is None or theta_d is None:
        return None, "zero_covariate_variance"
    adjusted = [
        (
            arm.mean[0] - theta_n * (arm.mean[x] - anchor),
            arm.mean[1] - theta_d * (arm.mean[x] - anchor),
        )
        for arm in (control, treatment)
    ]
    if not adjusted[0][1] or not adjusted[1][1]:
        return None, "zero_denominator_mean"
    parts = [
        _ratio(numerator, denominator, (theta_n, theta_d)) for numerator, denominator in adjusted
    ]
    return _anchored(parts[0], parts[1], n_c, n_t, covariate=True), None


def joint_means(
    control: GaussianState, treatment: GaussianState, declaration: ScalarMeanModel
) -> tuple[tuple[Fraction, Fraction] | None, str | None]:
    """``(f_c, f_t)`` at the observed means, or the reason no point exists."""
    linearisation, reason = linearise(control, treatment, declaration)
    if linearisation is None:
        return None, reason
    return (linearisation.control.value, linearisation.treatment.value), None


def _form(state: GaussianState, left: _Vector, right: _Vector) -> Fraction:
    """``left' S right / n^2``: the estimator-scale bilinear form of one arm."""
    total = _ZERO
    for i, a in enumerate(left):
        if a:
            total += a * sum((state.scatter[i][j] * b for j, b in enumerate(right) if b), _ZERO)
    return total / state.n**2


def _unavailable(alpha, alternative, count, reason) -> AsymptoticMeanSet:
    return AsymptoticMeanSet(
        components=(MeanSetComponent(lower=None, upper=None),),
        alpha=alpha,
        alternative=alternative,
        count=count,
        k=None,
        available=False,
        reason=reason,
    )


_Arms = tuple[tuple[str, GaussianState, _Arm], ...]


def _variance_forms(arms: _Arms) -> tuple[Fraction, Fraction, Fraction]:
    """``(slope, cross, constant)`` with ``V(r) = constant + 2 r cross + r^2 slope``.

    Each arm's contrast gradient is ``constant + r * slope`` and its scatter is
    symmetric, so these three forms are the whole of ``V`` at every ratio.
    """
    slope = sum((_form(s, arm.slope, arm.slope) for _, s, arm in arms), _ZERO)
    cross = sum((_form(s, arm.constant, arm.slope) for _, s, arm in arms), _ZERO)
    constant = sum((_form(s, arm.constant, arm.constant) for _, s, arm in arms), _ZERO)
    return slope, cross, constant


def _quadratic(
    forms: tuple[Fraction, Fraction, Fraction], d0: Fraction, d1: Fraction, k: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """Coefficients of ``D(r)^2 - k V(r)`` with ``D = d0 + d1 r`` and ``V`` from its forms."""
    slope, cross, constant = forms
    return d1 * d1 - k * slope, 2 * d0 * d1 - 2 * k * cross, d0 * d0 - k * constant


def _stability(arms: _Arms) -> tuple[bool, Fraction | None]:
    """Check positive adjusted denominators using pooled-anchor variance."""
    smallest: Fraction | None = None
    for index, (_, state, arm) in enumerate(arms):
        if arm.denominator is None:
            continue
        mean, gradient, cross = arm.denominator
        if mean <= 0:
            return False, None
        variance = _form(state, gradient, gradient)
        if cross is not None:
            variance += _form(arms[1 - index][1], cross, cross)
        if variance:
            precision = mean * mean / variance
            if smallest is None or precision < smallest:
                smallest = precision
    return True, smallest


@dataclass(frozen=True, slots=True)
class JointPreparation:
    """One stopped joint state's contrast before any error level exists.

    ``reason`` names a readiness or linearisation failure; it precedes every
    other reason and leaves the remaining fields at their empty values.
    ``forms`` are ``_variance_forms`` of the contrast gradients; ``contrast``
    and ``variance`` are ``D(r0)`` and ``V(r0)`` at the registered null ratio.
    ``degenerate`` is a nonpositive ``V(r0)`` or an arm whose own functional
    has zero variance. ``positive`` is False when a ratio denominator mean is
    not positive; ``stability`` is the smallest ratio-denominator
    ``Dbar^2 / Var(Dbar)`` of positive variance, None when nothing restricts.
    """

    count: int
    rho: Fraction
    reason: str | None = None
    control_value: Fraction = _ZERO
    treatment_value: Fraction = _ZERO
    forms: tuple[Fraction, Fraction, Fraction] = (_ZERO, _ZERO, _ZERO)
    contrast: Fraction = _ZERO
    variance: Fraction = _ZERO
    degenerate: bool = False
    positive: bool = True
    stability: Fraction | None = None

    def resolves(self, k: Fraction) -> bool:
        """Whether every ratio denominator's own count-clock sequence excludes
        zero at boundary ``k``: ``Dbar > 0`` and ``Dbar^2 > k Var(Dbar)``, the
        criterion the contrast itself is held to."""
        return self.positive and (self.stability is None or self.stability > k)

    def log_e(self, alternative: Alternative) -> Fraction | float:
        """Family evidence: the contrast's sign-gated mixture value, capped for a
        ratio law by its denominators' strict stability score.

        The cap reaches ``(-log_interval(alpha)).hi`` only where ``resolves``
        holds at ``count_boundary(count, alpha, rho)``, for every rational alpha;
        it reads no alpha, so a set unavailable at one level leaves this value
        unchanged. A state no level supports abstains.
        """
        if self.reason is not None or self.degenerate or not self.positive:
            return float("-inf")
        effect = count_boundary_log_e(
            count=self.count,
            rho=self.rho,
            estimator_contrast=self.contrast,
            estimator_variance=self.variance,
            alternative=alternative,
        )
        if self.stability is None:
            return effect
        return min(effect, _count_boundary_log_cap(self.count, self.rho, self.stability))


def prepare_joint(
    control: GaussianState,
    treatment: GaussianState,
    *,
    declaration: ScalarMeanModel,
    null_lift: Fraction,
) -> JointPreparation:
    """Screen, linearise and form one stopped joint state once, for every alpha."""
    dimension = retained_dimension(declaration.law)
    if len(control.mean) != dimension or len(treatment.mean) != dimension:
        refuse(_INVALID, reason=f"{declaration.law} requires {dimension}-dimensional arm moments")
    n = control.n + treatment.n
    reason = (
        "missing_arm"
        if not control.n or not treatment.n
        else "insufficient_arm_observations"
        if min(control.n, treatment.n) < 2
        else "before_declared_start"
        if min(control.n, treatment.n) < declaration.start_count
        else None
    )
    if reason is not None:
        return JointPreparation(n, declaration.rho, reason)
    linearisation, reason = linearise(control, treatment, declaration)
    if linearisation is None:
        return JointPreparation(n, declaration.rho, reason)
    arms = (
        ("control", control, linearisation.control),
        ("treatment", treatment, linearisation.treatment),
    )
    forms = _variance_forms(arms)
    slope, cross, constant = forms
    r0 = 1 + null_lift
    variance = constant + 2 * r0 * cross + r0 * r0 * slope
    positive, stability = _stability(arms)
    return JointPreparation(
        n,
        declaration.rho,
        control_value=linearisation.control.value,
        treatment_value=linearisation.treatment.value,
        forms=forms,
        contrast=linearisation.treatment.value - r0 * linearisation.control.value,
        variance=variance,
        degenerate=variance <= 0 or any(not _form(s, arm.own, arm.own) for _, s, arm in arms),
        positive=positive,
        stability=stability,
    )


def invert_joint(
    prepared: JointPreparation, *, alpha: Fraction, alternative: Alternative, e_value_dual: bool
) -> AsymptoticMeanSet:
    """Invert a prepared contrast at one error level's count-clock boundary.

    A look the declared contract cannot support yet -- a ratio denominator
    mean that is zero or not separated from zero at this look's boundary --
    is unavailable for this cell alone, exactly like a look before
    ``start_count``: no set, no decision, and no effect on sibling cells.
    """
    if not 0 < alpha < 1 or alternative not in ALTERNATIVE_VALUES:
        refuse(_INVALID, reason="alpha or alternative is outside the declared domain")
    level = boundary_alpha(alpha, alternative, e_value_dual=e_value_dual)
    n = prepared.count
    if prepared.reason is not None:
        return _unavailable(alpha, alternative, n, prepared.reason)
    k = count_boundary(n, level, prepared.rho)
    if not prepared.resolves(k):
        return _unavailable(alpha, alternative, n, "denominator_near_zero")
    if prepared.degenerate:
        return _unavailable(alpha, alternative, n, "zero_arm_variance")
    d0, d1 = prepared.treatment_value, -prepared.control_value
    return AsymptoticMeanSet(
        components=one_sided_components(
            _component_for_quadratic(*_quadratic(prepared.forms, d0, d1, k)),
            alternative,
            control_value=prepared.control_value,
            treatment_value=prepared.treatment_value,
        ),
        alpha=alpha,
        alternative=alternative,
        count=n,
        k=k,
        available=True,
        estimator_contrast=prepared.contrast,
        estimator_variance=prepared.variance,
    )


def asymptotic_joint_set(
    control: GaussianState,
    treatment: GaussianState,
    *,
    declaration: ScalarMeanModel,
    alpha: Fraction,
    null_lift: Fraction,
    alternative: Alternative,
    e_value_dual: bool = False,
) -> AsymptoticMeanSet:
    """Invert the linearised contrast at the count-clock boundary; see module notes."""
    return invert_joint(
        prepare_joint(control, treatment, declaration=declaration, null_lift=null_lift),
        alpha=alpha,
        alternative=alternative,
        e_value_dual=e_value_dual,
    )
