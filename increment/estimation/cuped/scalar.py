"""CUPED variance reduction -- one contrast-optimal theta, per-arm adjusted moments.

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces the variance
of an A/B test estimate by subtracting a pre-period covariate that is
correlated with the outcome but independent of treatment assignment.

Target
------
The adjustment exists to sharpen the ADDITIVE treatment-control contrast
``A - theta * B`` with ``A = mean(Y_t) - mean(Y_c)`` and
``B = mean(X_t) - mean(X_c)``. With independent arms and theta held fixed,

    Var(A - theta*B) = sum_a [Var(Y_a) - 2*theta*Cov(Y_a, X_a) + theta^2*Var(X_a)] / n_a

and the minimiser is the INVERSE-n weighted within-arm moment ratio

    theta = [Cov(Y_t,X_t)/n_t + Cov(Y_c,X_c)/n_c] / [Var(X_t)/n_t + Var(X_c)/n_c].

Each arm keeps its own centered moments (one shared slope, one intercept per
arm; Lin (2013)'s "ANCOVA2" form of covariate adjustment), so no between-arm
delta enters theta. The n-1 (pooled-regression) weights estimate a different
quantity -- the most precise pooled slope -- and under unequal allocation with
heterogeneous arm slopes can leave the contrast WORSE than unadjusted: the
accepted witness (n_t=900 at slope +10, n_c=100 at slope -1) inflated the
contrast variance 8.12x where this theta yields 1089/1090. Equal allocation
makes the two weightings coincide. For a family of more than two arms the
same theta minimises the SUM over every pair of that pair's additive contrast
variance; it does not optimise each heterogeneous pair on its own.

A per-arm theta reintroduces bias the other direction; the previous
JOINT-pooled theta (one mean across every arm, via ``ArmStats.combine()``)
reintroduces it too, via a between-arm cross term combine()'s ``cxy``
expansion adds (``n_arm * delta_x_arm * delta_y_arm``, nonzero whenever arms
differ in BOTH X and Y means -- which happens BECAUSE of the treatment effect
together with ordinary sampling imbalance in the covariate) that correlates
theta with the very contrast it is meant to be orthogonal to. The within-arm
form carries no such term by construction: verified by Monte Carlo (n=10/arm,
rho=0.9, 20000 reps) at a point-estimate bias of -2.6% of the true effect for
the joint form versus <0.1% for the within-arm form.

Pooled anchor and nonlinear projections
---------------------------------------
Each arm's adjusted mean is ``mean(Y_a) - theta * (mean(X_a) - mean_x_pooled)``
with ``mean_x_pooled`` the count-weighted covariate mean over every arm in
the fit (``ArmStats.combine()``, which does not carry the bias risk above).
That anchor is a RANDOM quantity shared by both arms. It cancels in an
additive difference, so the absolute contrast's variance is the ordinary
sum of per-arm adjusted variances (``CupedFit.adjust``). It does NOT cancel
in a log ratio or any other nonlinear projection: for anchor weights
``w_t = n_t/(n_t+n_c)``, ``w_c = 1 - w_t`` and a projection with Jacobian
``(j_t, j_c)`` in the adjusted means, the linearised arm scores are
``j_t*(Y_t - s_t*X_t)`` and ``j_c*(Y_c - s_c*X_c)`` with
``s_t = theta*k/j_t``, ``s_c = -theta*k/j_c``, ``k = j_t*w_c - j_c*w_t``
(``CupedFit.contrast_slopes``); the absolute projection ``(1, -1)`` gives
``k = 1`` and ``s_t = s_c = theta``. Consumers evaluate each arm's centered
quadratic form at its own slope (``CupedFit.residual_var``) rather than
summing marginal variances of two adjusted means that share the anchor.

What is and is not claimed
--------------------------
theta is chosen for additive precision; it is not universally optimal for
every nonlinear projection, whose uncertainty is nonetheless propagated
correctly above. theta is a nuisance fitted from the same finite sample and
then treated as fixed (the reported interval conditions on it), so the
variance statement is asymptotic (iid units, or the caller's declared
grain) and no finite-sample "never hurts" guarantee follows.

Ratio metrics
-------------
A ratio metric's estimand is ``E[Num]/E[Den]``, so the adjustment is
applied to each COMPONENT and the ratio is then formed from the adjusted
components (``fit_ratio_cuped``). The numerator and denominator each get
their OWN slope against the declared covariate -- ``theta_num`` from
``Cov(Num, X)/Var(X)`` and ``theta_den`` from ``Cov(Den, X)/Var(X)``, both
in the same inverse-n weighted within-arm form as above -- because the two
components track the covariate at different rates and one shared slope
would leave the better-predicted component under-adjusted.

The pooled anchor matters far more here than it does for a mean: without
it each adjusted component would estimate ``E[Num] - theta*E[X]`` rather
than ``E[Num]``, and their ratio would not be the ratio of anything. With
it, each adjusted component is consistent for its own expectation (X is
pre-period, so ``E[X_t] = E[X_c]``) and the adjusted ratio is consistent
for the SAME estimand as the unadjusted one.

The two adjusted components are correlated through the covariate they
share even where the raw components were not, so the ratio's variance
needs all three adjusted second moments::

    Var(Num - s_n*X)                = Var(Num) - 2*s_n*Cov(Num,X) + s_n^2*Var(X)
    Var(Den - s_d*X)                = Var(Den) - 2*s_d*Cov(Den,X) + s_d^2*Var(X)
    Cov(Num - s_n*X, Den - s_d*X)   = Cov(Num,Den) - s_d*Cov(Num,X)
                                      - s_n*Cov(Den,X) + s_n*s_d*Var(X)

Note which slope multiplies which cross moment: the numerator's own slope
``s_n`` rides on ``Cov(Den,X)``, not on ``Cov(Num,X)``. Dropping the
``s_n*s_d*Var(X)`` term (or the whole cross correction) understates the
ratio's variance whenever both components load on the covariate with the
same sign, which is the ordinary case. Those three moments are exactly
what the delta method's -2*Cov form consumes, so the adjusted ratio is
read through the same interval as an unadjusted one
(``variance.ratio_log_mean_se`` on the relative scale,
``variance.ratio_abs_diff_se`` on the absolute one) rather than a second
path of its own.

One covariate column serves both components; the per-component slopes are
what differ. Numerator and denominator adjusted against two DIFFERENT
pre-period columns is not expressible: an arm's moments carry a single
covariate family, so there is no second ``Cov(Num, Z)``/``Cov(Den, Z)``
pair to read. Adjust the two components as separate mean metrics for that.

References
----------
Deng et al. 2013. "Controlled-experiment Using Pre-Experiment Data."
    https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
Lin, W. 2013. "Agnostic notes on regression adjustments to experimental
    data: Reexamining Freedman's critique." Annals of Applied Statistics.
Deng et al. 2023. "From Augmentation to Decomposition: A New Look at
    CUPED in 2023." https://arxiv.org/abs/2312.02935 (Section 2.2
    identifies the joint-vs-within-arm distinction as ANCOVA1 vs
    ANCOVA2, and recommends ANCOVA2).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.armstats import (
    ArmStats,
    SummaryStats,
    assert_cauchy_schwarz,
    clamp_negative_variance,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.cuped.pooled_theta_least": "pooled_theta requires at least 2 arms (control + treatment)",
        "estimation.cuped.arm_no_covariate": "Arm {arm_group}/{arm_metric} has no covariate data (ref_x/cx1/cx2/cxy is None). The covariate must be materialised before requesting cuped: from a definitions path, set n_pre_periods > 0 on the experiment; from from_unit_summary, pass MetricSpec(name=..., covariate='<pre-period column>').",
        "estimation.cuped.covariate_zero_variance": "Covariate has zero variance (inverse-n weighted within-arm variance={weighted_var_x}). Cannot compute CUPED theta. Ensure the covariate is not constant; if it varies but sits at a large offset (|mean| >> spread), floating-point cancellation destroyed its variance before this seam -- centre the covariate (subtract a constant) before aggregation.",
        "estimation.cuped.cuped_adjusted_variance": "CUPED-adjusted variance for arm {arm_group!r}/{arm_metric!r} is {naive_var:.6g}, negative beyond floating-point rounding -- the slope implies an impossible {pair} correlation for this arm (Cauchy-Schwarz: {cross_label} cannot exceed sqrt(Var(Y)*Var(X))); the moments are corrupt or were not produced together",
        "estimation.cuped.ratio_arm_no_denominator_cross": "Arm {arm_group}/{arm_metric} carries a covariate and a ratio denominator but no cross moment between them (cxden is None), so the denominator's CUPED slope is not identified. Ratio CUPED needs Cov(covariate, denominator); supply moments in the centered (format-2) shape, which carries cxden, rather than the raw-sum (format-1) shape, which has no column for it.",
        "estimation.cuped.ratio_arm_covariate_role": "Arm {arm_group}/{arm_metric} declares x_role={x_role!r}, so its x family is not a CUPED covariate and its cross moment with the denominator means something else. Ratio CUPED needs x_role='covariate'.",
    },
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class _OutcomeFamily:
    """Which of an arm's outcome families a CUPED slope is fitted against.

    A mean metric has one (``y``); a ratio metric has two (``y`` carries the
    numerator, ``den`` the denominator), each adjusted against the same
    covariate with its own slope.
    """

    mean: Callable[[ArmStats], float]
    var: Callable[[ArmStats], float]
    cross: Callable[[ArmStats], float]
    #: How this family's cross moment with the covariate reads in a refusal.
    cross_label: str
    #: How this family pairs with the covariate in a refusal.
    pair_label: str


NUMERATOR = _OutcomeFamily(
    mean=ArmStats.mean_y,
    var=ArmStats.var_y,
    cross=ArmStats.cov_yx,
    cross_label="Cov(Y,X)",
    pair_label="Y/X",
)
DENOMINATOR = _OutcomeFamily(
    mean=ArmStats.mean_den,
    var=ArmStats.var_den,
    cross=ArmStats.cov_xden,
    cross_label="Cov(Den,X)",
    pair_label="Den/X",
)


@dataclass(frozen=True, slots=True)
class CupedFit:
    """One family's fitted CUPED nuisance: theta, the pooled anchor and the
    arms it was fitted on, computed once by :func:`fit_cuped` and read by
    every projection so a consumer never re-estimates them.

    ``arms`` keeps the caller's order; every method takes the arm object
    itself so joint (Y, X) moments are read from the original record rather
    than carried as a second copy. ``family`` selects which outcome the
    slope was fitted against -- the ratio denominator reads the same
    covariate through its own moments.
    """

    theta: float
    mean_x_pooled: float
    arms: tuple[ArmStats, ...]
    family: _OutcomeFamily = NUMERATOR

    def adjusted_mean(self, arm: ArmStats) -> float:
        """``mean(Y) - theta * (mean(X) - mean_x_pooled)`` for *arm*."""
        return self.family.mean(arm) - self.theta * (arm.mean_x() - self.mean_x_pooled)

    def residual_var(self, arm: ArmStats, slope: float) -> float:
        """ddof=1 variance of ``Y - slope * X`` on *arm*: the full centered
        quadratic form ``Var(Y) - 2*slope*Cov(Y,X) + slope^2*Var(X)``, never
        the ``(1 - rho^2)`` shortcut (exact only at the arm's own slope).

        A deficit inside the rounding tolerance of its constituent terms is
        cancellation and clamps to zero; a material deficit means the moments
        were not produced together and is refused by name.
        """
        var_y = self.family.var(arm)
        cross_product = slope * self.family.cross(arm)
        cross = cross_product + cross_product
        square_scale = slope * math.sqrt(arm.var_x())
        square = square_scale * square_scale
        naive_var = var_y - cross + square
        magnitude = abs(var_y) + abs(cross) + abs(square)
        adjusted_var = clamp_negative_variance(naive_var, magnitude=magnitude, n=arm.n)
        if adjusted_var is None:
            _raise(
                "estimation.cuped.cuped_adjusted_variance",
                arm_group=arm.group_id,
                arm_metric=arm.metric,
                naive_var=naive_var,
                pair=self.family.pair_label,
                cross_label=self.family.cross_label,
            )
        return adjusted_var

    def adjust(self) -> list[SummaryStats]:
        """Per-arm adjusted mean and ``Var(Y - theta*X)``, in ``arms`` order.

        These are the absolute projection's pieces: the anchor cancels in a
        difference of two adjusted means, so ``sum(var/n)`` over a pair is
        that pair's additive contrast variance.
        """
        return [
            SummaryStats(
                n=arm.n, mean=self.adjusted_mean(arm), var=self.residual_var(arm, self.theta)
            )
            for arm in self.arms
        ]

    def contrast_slopes(
        self, treatment: ArmStats, control: ArmStats, j_t: float, j_c: float
    ) -> tuple[float, float]:
        """``(s_t, s_c)``: the per-arm slopes of a projection whose Jacobian
        in the adjusted means ``(mu_t, mu_c)`` is ``(j_t, j_c)``.

        Differentiating ``mu_a = mean(Y_a) - theta*(mean(X_a) - anchor)``
        through the shared anchor ``w_t*mean(X_t) + w_c*mean(X_c)`` gives
        the linearised contrast ``j_t*(Y_t - s_t*X_t)`` on treatment units and
        ``j_c*(Y_c - s_c*X_c)`` on control units with ``k = j_t*w_c - j_c*w_t``,
        ``s_t = theta*k/j_t`` and ``s_c = -theta*k/j_c``. The consumer's per-arm
        variance is then ``j_a^2 * residual_var(arm_a, s_a) / n_a``, which keeps
        the ``1/mu`` factor of a log Jacobian on the consumer's own
        magnitude-aware scale. Defined for a fit over exactly this pair; both
        Jacobian entries must be nonzero.
        """
        assert len(self.arms) == 2 and treatment is not control, (
            "unreachable: contrast projections are fitted on exactly the pair"
        )
        assert any(a is treatment for a in self.arms) and any(a is control for a in self.arms), (
            "unreachable: contrast projections read the arms the fit was built on"
        )
        assert j_t != 0.0 and j_c != 0.0, "unreachable: a contrast Jacobian names both arms"
        n = treatment.n + control.n
        k = j_t * (control.n / n) - j_c * (treatment.n / n)
        return self.theta * k / j_t, -self.theta * k / j_c


def fit_cuped(arms: list[ArmStats], family: _OutcomeFamily = NUMERATOR) -> CupedFit:
    """Fit theta and the pooled anchor once for *arms* (control + treatment,
    or a family). See the module docstring for the objective theta minimises.

    *family* selects the outcome the slope is fitted against; a ratio metric
    fits one :class:`CupedFit` per component against the same covariate.

    Raises ``InvalidRequestError`` on fewer than 2 arms, un-materialised
    covariate moments, a within-arm covariate variance that is zero in every
    arm, or an arm whose own moments violate Cauchy-Schwarz.
    """
    theta = _within_arm_theta(arms, family)
    # Pooling arms is a partition combination: combine() re-centers every arm
    # on the pooled reference, so no large-mean term forms. Used only for the
    # centering anchor, not theta itself, so it does not carry the between-arm
    # bias theta must avoid.
    pooled = ArmStats.combine(arms, group_id="(pooled)")
    return CupedFit(theta=theta, mean_x_pooled=pooled.mean_x(), arms=tuple(arms), family=family)


def pooled_theta(arms: list[ArmStats]) -> tuple[float, float]:
    """``(theta, mean_x_pooled)`` of :func:`fit_cuped` for callers that only
    need the nuisance values."""
    fit = fit_cuped(arms)
    return fit.theta, fit.mean_x_pooled


def _within_arm_theta(arms: list[ArmStats], family: _OutcomeFamily = NUMERATOR) -> float:
    """The inverse-n weighted within-arm moment ratio."""
    if len(arms) < 2:
        _raise("estimation.cuped.pooled_theta_least")

    for arm in arms:
        if arm.ref_x is None or arm.cx1 is None or arm.cx2 is None or arm.cxy is None:
            _raise(
                "estimation.cuped.arm_no_covariate",
                arm_group=arm.group_id,
                arm_metric=arm.metric,
            )

    # Within-arm moments weighted by 1/n minimize the additive contrast variance, so no
    # cross-arm delta enters theta. Use corrected ddof=1 moments, not raw cxy/cx2 centered on a
    # stored reference: raw sums bias theta when that reference is offset (~17% at one
    # covariate SD) and disagree with the adjusted variance.
    weighted_cov = 0.0
    weighted_var_x = 0.0
    for arm in arms:
        var_x = arm.var_x()
        if var_x <= 0.0:
            # A constant covariate within an arm identifies nothing about theta. Drop its
            # cross moment with its zero variance: feasibility tolerates a cross moment inside
            # the zero-variance rounding floor, which would move theta with no denominator.
            continue
        weighted_cov += family.cross(arm) / arm.n
        weighted_var_x += var_x / arm.n

    # Check each arm's own Cauchy-Schwarz bound before pooling: at the shared theta, equal
    # variances with opposite impossible covariances cancel to theta == 0 and pass the
    # downstream adjusted-variance check.
    for arm in arms:
        degrees = arm.n - 1
        assert_cauchy_schwarz(
            degrees * family.cross(arm),
            degrees * family.var(arm),
            degrees * arm.var_x(),
            arm.n,
            what=(
                f"CUPED covariate moments for arm {arm.group_id!r}/{arm.metric!r}: "
                f"{family.cross_label}"
            ),
        )

    if weighted_var_x <= 0:
        _raise("estimation.cuped.covariate_zero_variance", weighted_var_x=weighted_var_x)
    return weighted_cov / weighted_var_x


def cuped_adjust(arms: list[ArmStats]) -> list[SummaryStats]:
    """Apply CUPED variance reduction to a matched pair (or family) of arms:
    :meth:`CupedFit.adjust` of :func:`fit_cuped`.

    Returns per-arm :class:`SummaryStats` with the adjusted mean and
    ``Var(Y - theta*X)`` (full quadratic form, not the ``(1-rho^2)``
    shortcut). *arms* must carry materialised covariate moments
    (``ref_x``/``cx1``/``cx2``/``cxy`` not None).

    Raises ``InvalidRequestError`` if any arm has un-materialised covariate
    fields, if the within-arm covariate variance is zero in every arm, if
    fewer than 2 arms are provided, or if theta implies an impossible Y/X
    correlation (Cauchy-Schwarz: |Cov(Y,X)| <= sqrt(Var(Y)*Var(X))) for any
    individual arm's adjusted variance -- refused rather than silently
    clamped to a plausible-looking zero.
    """
    return fit_cuped(arms).adjust()


@dataclass(frozen=True, slots=True)
class AdjustedRatioMoments:
    """One arm's CUPED-adjusted ratio moments, in the order the delta-method
    ratio reductions take them (``variance.ratio_log_mean_se`` and
    ``variance.ratio_abs_diff_se``).

    ``cov_num_den`` is the covariance of the two ADJUSTED components, which
    carries the correlation the shared covariate induces between them; it is
    not the raw ``Cov(Num, Den)``.
    """

    num_bar: float
    den_bar: float
    var_num: float
    var_den: float
    cov_num_den: float
    n: int


@dataclass(frozen=True, slots=True)
class RatioCupedFit:
    """A ratio metric's two CUPED slopes against one shared covariate: one
    fitted on the numerator, one on the denominator, sharing the pooled
    anchor and the arms they were fitted on.

    See the module docstring for why each component needs its own slope and
    for the adjusted second moments this produces.
    """

    numerator: CupedFit
    denominator: CupedFit

    @property
    def arms(self) -> tuple[ArmStats, ...]:
        return self.numerator.arms

    def adjusted_components(self, arm: ArmStats) -> tuple[float, float]:
        """``(adjusted numerator mean, adjusted denominator mean)`` for *arm*.

        Both are anchored on the pooled covariate mean, so each estimates its
        own component's expectation and their ratio estimates the SAME
        estimand the unadjusted ratio does.
        """
        return self.numerator.adjusted_mean(arm), self.denominator.adjusted_mean(arm)

    def relative_moments(
        self, treatment: ArmStats, control: ArmStats
    ) -> tuple[AdjustedRatioMoments, AdjustedRatioMoments]:
        """``(treatment, control)`` adjusted moments for ``log(R_t / R_c)``,
        whose per-arm Jacobian in the adjusted components is
        ``(1/num_bar, -1/den_bar)``."""
        return self._projected(treatment, control, _log_ratio_jacobian)

    def absolute_moments(
        self, treatment: ArmStats, control: ArmStats
    ) -> tuple[AdjustedRatioMoments, AdjustedRatioMoments]:
        """``(treatment, control)`` adjusted moments for ``R_t - R_c``, whose
        per-arm Jacobian in the adjusted components is
        ``(1/den_bar, -num_bar/den_bar**2)`` -- the log Jacobian scaled by
        ``R``, so the two projections coincide where the arms' ratios do."""
        return self._projected(treatment, control, _absolute_ratio_jacobian)

    def _projected(
        self,
        treatment: ArmStats,
        control: ArmStats,
        jacobian: Callable[[float, float], tuple[float, float]],
    ) -> tuple[AdjustedRatioMoments, AdjustedRatioMoments]:
        """Adjusted moments at the per-arm slopes this projection implies.

        The pooled anchor is shared by both arms, so it does not cancel in a
        ratio the way it does in an additive difference: each component's
        linearised slope picks up the other arm through the anchor. That is
        the same correction :meth:`CupedFit.contrast_slopes` makes for a mean
        metric, applied once per component with that component's own
        Jacobian, and it collapses to the fitted thetas when the two arms'
        adjusted ratios agree.
        """
        num_t, den_t = self.adjusted_components(treatment)
        num_c, den_c = self.adjusted_components(control)
        a_t, b_t = jacobian(num_t, den_t)
        a_c, b_c = jacobian(num_c, den_c)
        slope_num_t, slope_num_c = self.numerator.contrast_slopes(treatment, control, a_t, -a_c)
        slope_den_t, slope_den_c = self.denominator.contrast_slopes(treatment, control, -b_t, b_c)
        return (
            self._arm_moments(treatment, num_t, den_t, slope_num_t, slope_den_t),
            self._arm_moments(control, num_c, den_c, slope_num_c, slope_den_c),
        )

    def at_fitted_slopes(self, arm: ArmStats) -> AdjustedRatioMoments:
        """*arm*'s adjusted moments at the fitted slopes themselves: what
        :meth:`_projected` reduces to when the two arms' adjusted ratios agree,
        and so the per-arm variance a planning baseline projects forward."""
        num_bar, den_bar = self.adjusted_components(arm)
        return self._arm_moments(
            arm, num_bar, den_bar, self.numerator.theta, self.denominator.theta
        )

    def _arm_moments(
        self,
        arm: ArmStats,
        num_bar: float,
        den_bar: float,
        slope_num: float,
        slope_den: float,
    ) -> AdjustedRatioMoments:
        return AdjustedRatioMoments(
            num_bar=num_bar,
            den_bar=den_bar,
            var_num=self.numerator.residual_var(arm, slope_num),
            var_den=self.denominator.residual_var(arm, slope_den),
            cov_num_den=_adjusted_cross_cov(arm, slope_num, slope_den),
            n=arm.n,
        )


def _log_ratio_jacobian(num_bar: float, den_bar: float) -> tuple[float, float]:
    return 1.0 / num_bar, 1.0 / den_bar


def _absolute_ratio_jacobian(num_bar: float, den_bar: float) -> tuple[float, float]:
    return 1.0 / den_bar, (num_bar / den_bar) / den_bar


def _adjusted_cross_cov(arm: ArmStats, slope_num: float, slope_den: float) -> float:
    """``Cov(Num - s_n*X, Den - s_d*X)`` on *arm*.

    Expanding the bilinear form puts the DENOMINATOR's slope on
    ``Cov(Num, X)`` and the NUMERATOR's on ``Cov(Den, X)``; the final
    ``s_n*s_d*Var(X)`` term is the correlation the shared covariate induces
    between two components that may have had none. Dropping any of the three
    corrections understates the ratio's variance whenever both components
    load on the covariate with the same sign.

    Unlike a variance there is no sign constraint to police here: the
    reductions that consume it (``variance.ratio_log_mean_se`` and
    ``variance.ratio_abs_diff_se``) judge the whole quadratic form, which is
    where an infeasible cross moment actually shows up.
    """
    num_mantissa, num_exponent = math.frexp(slope_num)
    den_mantissa, den_exponent = math.frexp(slope_den)
    var_mantissa, var_exponent = math.frexp(arm.var_x())
    mantissa = num_mantissa * den_mantissa * var_mantissa
    try:
        shared_covariance = math.ldexp(mantissa, num_exponent + den_exponent + var_exponent)
    except OverflowError:
        shared_covariance = math.copysign(math.inf, mantissa)
    return (
        arm.cov_yden() - slope_den * arm.cov_yx() - slope_num * arm.cov_xden() + shared_covariance
    )


def fit_ratio_cuped(arms: list[ArmStats]) -> RatioCupedFit:
    """Fit a ratio metric's numerator and denominator slopes against the one
    declared covariate, sharing a single pooled anchor.

    Raises ``InvalidRequestError`` on everything :func:`fit_cuped` refuses,
    plus an arm whose x family is not a CUPED covariate or which carries no
    cross moment between the covariate and the denominator -- the term the
    adjusted components' covariance needs, and which cannot be inferred.
    """
    # The numerator slope first: it refuses a missing covariate family by
    # name, so the ratio-specific checks below never run on an arm that has
    # no covariate at all and would fail them for the wrong reason.
    theta_num = _within_arm_theta(arms, NUMERATOR)
    for arm in arms:
        if not arm.moments.has("x"):
            _raise(
                "estimation.cuped.ratio_arm_covariate_role",
                arm_group=arm.group_id,
                arm_metric=arm.metric,
                x_role=arm.x_role,
            )
        if arm.cxden is None:
            _raise(
                "estimation.cuped.ratio_arm_no_denominator_cross",
                arm_group=arm.group_id,
                arm_metric=arm.metric,
            )
    theta_den = _within_arm_theta(arms, DENOMINATOR)
    anchor = ArmStats.combine(arms, group_id="(pooled)").mean_x()
    frozen = tuple(arms)
    return RatioCupedFit(
        numerator=CupedFit(theta_num, anchor, frozen, NUMERATOR),
        denominator=CupedFit(theta_den, anchor, frozen, DENOMINATOR),
    )
