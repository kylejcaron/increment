"""Sitewide impact: translate an exposed-population lift into a whole-site number.

Pure math over ``ArmStats`` - no ibis, no warehouse access. The query layer
computes the site-window volume and threads it in as ``site_total_volume``.
Sum-type metrics go through ``sitewide_impact``; ratio metrics through
``sitewide_impact_ratio``. Per-unit-window metrics (retention, quantile) are
out of scope for both.

The math
--------
Given a per-unit absolute lift ``delta = target.mean - control.mean``, its
Welch SE ``se = sqrt(target.var_mean + control.var_mean)``, and the observed
site-window total ``V`` (everything on the metric's raw event stream over
the enrollment window, not just the exposed population):

* **Counterfactual baseline** ``V0 = V - delta * n_T``: what the site would
  have produced with nobody exposed to treatment.
* **Ship-to-all impact** ``I = delta * N_exp`` (``N_exp = n_C + n_T``): the
  absolute site-wide gain from rolling the lift out to every enrolled unit.
* **Relative impact** ``I / V0``: the ship-to-all gain as a fraction of the
  counterfactual baseline.

``V`` and ``N_exp`` are DATA, not estimated quantities - fixed constants for
the delta-method CI below, which propagates only ``delta``'s sampling
variance.

.. warning::

   The baseline assumes enrolled units exhaust the treated population.
   ``V0`` nets out only enrolled units, exact under unit randomization but
   biased under cluster randomization (a treated unit outside any exposure
   row is still treated but invisible to ``n_i``): relative impact is then
   a lower bound on the true value; absolute impact is unaffected.

Absolute impact is linear in delta, so its CI is the lift's Wald interval
scaled by ``N_exp``. Relative impact is not - ``V0`` is itself a function of
delta - so its derivative needs the quotient rule::

    f(delta) = delta * N_exp / (V - delta * n_T)
             = u(delta) / v(delta),   u = delta * N_exp,   v = V - delta * n_T

    f'(delta) = (u'v - u*v') / v**2
              = (N_exp*(V - delta*n_T) - delta*N_exp*(-n_T)) / V0**2
              = (N_exp*V - N_exp*delta*n_T + N_exp*delta*n_T) / V0**2
              = N_exp * V / V0**2

The ``delta*n_T`` cross-terms cancel exactly: ``se(relative_impact) =
|N_exp * V / V0**2| * se(delta)``, the delta-method (first-order Taylor) SE.

The ratio math
--------------
Ratio metrics need two per-unit lifts: the numerator lift ``delta_num =
target.mean - control.mean`` and denominator lift ``delta_den =
target.mean_den - control.mean_den``, with Welch variances::

    Var(delta_num) = target.var_mean + control.var_mean
    Var(delta_den) = target.var_mean_den + control.var_mean_den

and a covariance term, since both are computed from the same per-arm sums::

    Cov(delta_num, delta_den) = target.cov_mean_den + control.cov_mean_den

Given the site-window totals ``V_num``, ``V_den`` and ``N_exp = n_C + n_T``::

    N0 = V_num - n_T * delta_num       D0 = V_den - n_T * delta_den
    N1 = N0 + N_exp * delta_num        D1 = D0 + N_exp * delta_den

and impact is the difference of the two ratios::

    impact = N1/D1 - N0/D0

The partials (``g_num``, ``g_den``) follow the quotient/chain rule, no cross
terms::

    d(impact)/d(delta_num) = n_C/D1 + n_T/D0
    d(impact)/d(delta_den) = -N1*n_C/D1**2 - N0*n_T/D0**2

Its variance includes the numerator/denominator covariance term::

    Var(impact) = g_num**2 * Var(delta_num) + g_den**2 * Var(delta_den)
                + 2 * g_num * g_den * Cov(delta_num, delta_den)

Relative impact is absolute impact as a fraction of ``R0 = N0/D0``::

    rel = A(delta_num) * B(delta_den) - 1,   A = N1/N0,   B = D0/D1

    h_num = B * dA/d(delta_num) = (D0/D1) * (n_C*N0 + n_T*N1) / N0**2
    h_den = A * dB/d(delta_den) = -(N1/N0) * (n_T*D1 + n_C*D0) / D1**2

    Var(rel) = h_num**2 * Var(delta_num) + h_den**2 * Var(delta_den)
             + 2 * h_num * h_den * Cov(delta_num, delta_den)

``rel`` also requires ``N0 > 0`` (a zero or negative counterfactual ratio
makes "fraction of baseline" meaningless), so both refuse rather than
return a nonsense sign.

Multi-arm experiments
----------------------
With a co-enrolled arm, every non-control arm's lift is netted out of the
baseline (``V0 = V - sum_i(delta_i * n_i)``, ``N_exp = n_C + sum_i n_i``);
omitting ``other_arms`` recovers the single-arm formulas term for term. The
reported number is always "target ships to everyone, no other arm ever
ran" against "nobody ships" - the only two worlds a ship decision can
actually choose between.

**Cross-arm covariance.** Every ``delta_i = mean_i - mean_C`` shares the
same control mean, so for ``i != j``::

    Cov(delta_i, delta_j) = control.var_mean

Expanding the gradient contraction back into independent arm means avoids
assembling the full covariance matrix::

    Var(...) = sum_i g_i**2 * var_mean(i) + (sum_i g_i)**2 * control.var_mean

:func:`_combination_var` returns this sum's terms, not their total (see
"Degrees of freedom" below for why).

Degrees of freedom
-------------------
At iid grain both functions use the Normal reference. At cluster grain the
critical value is ``t_{dof}``, with Welch-Satterthwaite degrees of freedom from
each contributing arm's own Bessel-corrected between-cluster variance, never a
pooled ``K - 2`` reference.

* **Sum-metric absolute impact** is linear in the target delta alone
  (``delta_se = sqrt(target.var_mean + control.var_mean)``), so its dof
  is the plain two-component Welch-Satterthwaite reduction over
  ``(target.var_mean, target.own_dof)`` and ``(control.var_mean,
  control.own_dof)`` -- reducing exactly to the pairwise ``K_target +
  K_control - 2`` at equal arm count/variance, correctly widening under
  imbalance or heteroskedastic arm variance where that pairwise value
  cannot.
* **Relative impact (both families) and ratio-metric absolute impact** mix
  every arm's contribution with different weights, so a pooled dof
  overstates precision. These use a Welch-Satterthwaite reduction over
  :func:`_combination_var`'s per-term variance components ``v_i``::

      dof = (sum(v_i))**2 / sum(v_i**2 / dof_i)

  where ``dof_i`` is each component's own ``K_i - 1`` (``_Arm.own_dof``):
  an arm's own cluster count minus 1 for its term, and the control's own
  cluster count minus 1 for the shared control-anchored term. Each term is
  a single arm's sample variance, so crediting it with any other arm's
  clusters (the pooled ``n_clusters - 2``) double-counts degrees and can
  lift the reduction above the bound ``sum_i(K_i - 1)`` it can never
  exceed. At iid grain every ``_Arm.own_dof`` is ``None`` and this
  reduction is skipped for the Normal reference.

``SitewideImpact``/``SitewideRatioImpact`` expose the dof each interval
above actually used as ``absolute_dof``/``relative_dof`` - not the
contrast's pooled ``n_clusters - 2``, which only coincides with either
of them (in the sum-metric case, only approximately, at equal arm
count/variance) when no other arm is enrolled.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator
from scipy.stats import norm as _norm

from increment._moment_plan import X_SLOT_ROLES
from increment.errors import (
    CodedModel,
    IncrementWarning,
    InvalidRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.armstats import (
    ARM_STATS_CROSS_FIELD_FINITE,
    ArmStats,
    welch_satterthwaite_df,
)
from increment.estimation.engine import check_total_clusters, ratio_abs_diff_se, ratio_pair_cov
from increment.estimation.variance import cluster_outcome_moments, ratio_moments

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.sitewide.control_carries_metric": "control carries metric {control!r} but target carries {target!r} -- a contrast is one metric, never two",
        "estimation.sitewide.control_carries_study": "control carries study_id {control!r} but target carries {target!r} -- a contrast is one experiment, never two",
        "estimation.sitewide.control_target_are": "control and target are both group {control!r} -- a contrast needs two distinct arms",
        "estimation.sitewide.other_arms_entry_metric_mismatch": "other_arms entry {arm!r} carries metric {metric!r} but the contrast is on {target!r} -- netting another metric's lift out of the site total would corrupt the counterfactual baseline",
        "estimation.sitewide.other_arms_entry_study_mismatch": "other_arms entry {arm!r} carries study_id {study_id!r} but the contrast is on study_id {target_study_id!r} -- co-enrolling an arm from a different experiment would corrupt the counterfactual baseline",
        "estimation.sitewide.other_arms_repeats": "other_arms repeats group {arm!r} (already the control, the target arm, or an earlier other_arms entry) -- its lift would leave the counterfactual baseline twice",
        "estimation.sitewide.from_iid.metric_arm_carries_ref_den": "metric {target!r}: {label} arm {arm!r} carries a populated ref_den, which looks like a clustered row (per-cluster size rides the den family for a sum metric) -- build this contrast with SitewideContrast.from_clusters, not from_iid.",
        "estimation.sitewide.sitewide_ratio.metric_arm_missing": "metric {metric!r}: arm {arm!r} is missing the denominator moments a ratio contrast requires -- build it through SitewideRatioContrast.from_iid/from_clusters, never from a sum-family arm.",
        "estimation.sitewide.sitewide_ratio.metric_arm_declares": "metric {target!r}: {label} arm {arm!r} declares x_role={x_role!r}, which is a cluster-grain row -- {remedy}.",
        "estimation.sitewide.alpha": "alpha must be in (0, 1), got {alpha!r}",
        "estimation.sitewide.counterfactual_baseline_volume": "counterfactual baseline volume must be positive, got {baseline_volume!r} (site_total_volume={site_total_volume!r}, delta={delta!r}, target.n_units={n_units!r}, other arms {other_arms!r} contributing {other_contribution!r}) -- relative impact is undefined against a non-positive baseline",
        "estimation.sitewide.counterfactual_baseline_denominator": "counterfactual baseline denominator must be positive, got {baseline_denominator!r} (site_total_denominator={site_total_denominator!r}, delta_den={delta_den!r}, target.n_units={n_units!r}, other arms {other_arms!r}, total enrolled denominator lift {den_contribution!r}) -- the ratio metric is undefined against a non-positive denominator",
        "estimation.sitewide.ship_all_denominator": "ship-to-all denominator must be positive, got {shipped_denominator!r} (baseline_denominator={baseline_denominator!r}, delta_den={delta_den!r}, n_exp={n_exp!r}) -- the ratio metric is undefined against a non-positive denominator",
        "estimation.sitewide.cluster_arm_needs_two": "metric '{metric}': cluster contrast ('{cluster}') needs at least 2 clusters in EVERY enrolled arm to estimate a between-cluster variance for that arm, got {arm_counts!r} -- a single cluster carries no between-cluster variance contribution to estimate its own arm's Welch-Satterthwaite component from.",
        "estimation.sitewide.counterfactual_baseline_ratio": "counterfactual baseline ratio must be positive, got numerator {baseline_numerator!r} over denominator {baseline_denominator!r} (site_total_numerator={site_total_numerator!r}, delta_num={delta_num!r}, target.n_units={n_units!r}, other arms {other_arms!r}, total enrolled numerator lift {num_contribution!r}) -- relative impact is undefined against a non-positive baseline",
    },
)

_REFUSALS["estimation.armstats.arm_stats.cross_field_finite"] = ARM_STATS_CROSS_FIELD_FINITE
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(code: str, warning_type: type[IncrementWarning], render) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    # +1 compensates for this helper's own frame; errors.warn() adds its
    # own +1 internally for its frame, so the numeric stacklevel literal at
    # each call site keeps its pre-conversion meaning.
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "estimation.sitewide.cluster_baseline_assumption",
    IncrementWarning,
    lambda: _CLUSTER_BASELINE_CAVEAT,
)

# Emitted on the clustered sitewide path: a treated unit with no exposure
# row is invisible to the enrolled count, so relative impact is a lower bound.
_CLUSTER_BASELINE_CAVEAT = (
    "sitewide under a declared cluster: the whole-site baseline assumes the "
    "enrolled units exhaust the treated population. Under cluster "
    "randomization a unit inside a treated cluster that produced no exposure "
    "row is still treated -- its volume is in the observed site total but not "
    "the enrolled count. For a positive lift the counterfactual baseline is "
    "biased upward and relative impact downward; in general the reported "
    "relative impact is a lower bound on the true value. The point estimate of "
    "absolute impact is unaffected; treat relative impact as a lower bound."
)


def _warn_cluster_baseline_assumption() -> None:
    """Surface the enrolled-exhausts-treated assumption on the clustered
    sitewide path (see ``_CLUSTER_BASELINE_CAVEAT``)."""
    _warn("estimation.sitewide.cluster_baseline_assumption", stacklevel=3)


class _Arm(BaseModel):
    """One arm's per-unit moments, normalized to the same shape whether it
    came from iid units or cluster sums.

    ``own_dof`` is this arm's cluster count minus 1 (``None`` at iid grain).
    Every Welch-Satterthwaite reduction pairs each variance term with the
    degrees of freedom of the arm that estimated it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    n_units: float
    mean: float
    var_mean: float
    mean_den: float | None = None
    var_mean_den: float | None = None
    cov_mean_den: float | None = None
    own_dof: float | None = None


class SitewideImpact(BaseModel):
    """Whole-site impact of shipping a lift to every enrolled unit.

    Parameters
    ----------
    delta, delta_se : float
        Per-unit absolute lift of the target arm and its standard error.
    treatment_group : str
        ``group_id`` of the target arm.
    n_control, n_treatment : float
        Control and target arm population sizes (float since a cluster
        grain population is ``K * mean cluster size``, not an integer).
    n_enrolled : float
        Every enrolled unit across all arms (``N_exp``); equals
        ``n_control + n_treatment`` only when no other arm is enrolled.
    other_arm_ids : tuple[str, ...]
        ``group_id`` of every other enrolled non-control arm, whose lift
        was netted out of ``baseline_volume``.
    site_total_volume : float
        The observed site-window total this result was computed against.
    baseline_volume : float
        Counterfactual site-window total with nobody exposed to treatment.
    absolute_impact, absolute_impact_se, absolute_impact_lb, absolute_impact_ub : float
        Ship-to-all absolute impact, its SE, and its ``alpha``-level interval.
    relative_impact, relative_impact_se, relative_impact_lb, relative_impact_ub : float
        Ship-to-all impact as a fraction of ``baseline_volume``, its SE,
        and its ``alpha``-level interval.
    alpha : float
        Two-sided significance level the intervals were built at.
    n_clusters : int | None
        ``None`` at iid grain, else the contrast's total cluster count.
    absolute_dof, relative_dof : float | None
        ``None`` at iid grain (the Normal reference applies). Otherwise
        the dof ``absolute_impact``'s and ``relative_impact``'s own
        critical values were cut at - see "Degrees of freedom" in the
        module docstring for which reduction each is. They differ from
        each other, and both differ from the contrast's pooled
        ``n_clusters - 2`` once another arm is enrolled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: float
    delta_se: float
    treatment_group: str
    n_control: float
    n_treatment: float
    n_enrolled: float
    other_arm_ids: tuple[str, ...]
    site_total_volume: float
    baseline_volume: float
    absolute_impact: float
    absolute_impact_se: float
    absolute_impact_lb: float
    absolute_impact_ub: float
    relative_impact: float
    relative_impact_se: float
    relative_impact_lb: float
    relative_impact_ub: float
    alpha: float
    n_clusters: int | None = None
    absolute_dof: float | None = None
    relative_dof: float | None = None


class SitewideRatioImpact(BaseModel):
    """Whole-site impact of shipping a ratio metric's lift to every unit.

    Ratio metrics need two per-unit lifts (numerator, denominator) instead
    of :class:`SitewideImpact`'s single ``delta``, hence a separate model.
    See "The ratio math" in the module docstring for the derivation.

    Parameters
    ----------
    delta_num, delta_num_se : float
        Per-unit absolute lift of the target arm's numerator and its SE.
    delta_den, delta_den_se : float
        Per-unit absolute lift of the target arm's denominator and its SE.
    delta_cov : float
        Covariance of ``delta_num`` and ``delta_den``.
    treatment_group : str
        ``group_id`` of the target arm.
    n_control, n_treatment : float
        Control and target arm population sizes (see
        :class:`SitewideImpact` for why this is a float).
    n_enrolled : float
        Every enrolled unit across all arms (``N_exp``).
    other_arm_ids : tuple[str, ...]
        ``group_id`` of every other enrolled non-control arm, whose
        numerator and denominator lifts were netted out of the baselines.
    site_total_numerator, site_total_denominator : float
        The observed site-window totals this result was computed against.
    baseline_numerator, baseline_denominator, baseline_ratio : float
        Counterfactual site-window totals with nobody exposed to
        treatment (``N0``, ``D0``), and their ratio.
    shipped_numerator, shipped_denominator, shipped_ratio : float
        Ship-to-all site-window totals (``N1``, ``D1``), and their ratio.
    absolute_impact, absolute_impact_se, absolute_impact_lb, absolute_impact_ub : float
        Ship-to-all absolute impact (``shipped_ratio - baseline_ratio``),
        its delta-method SE, and its ``alpha``-level Wald interval.
    relative_impact, relative_impact_se, relative_impact_lb, relative_impact_ub : float
        Ship-to-all impact as a fraction of ``baseline_ratio``, its
        delta-method SE, and its ``alpha``-level interval.
    alpha : float
        Two-sided significance level the interval was built at.
    n_clusters, absolute_dof, relative_dof : int | None, float | None, float | None
        Same contract as :class:`SitewideImpact`'s fields of the same
        name - unlike the sum-metric's ``absolute_dof``, this class's
        ``absolute_dof`` is always the Satterthwaite reduction, never
        the plain pairwise dof (see "Degrees of freedom" in the module
        docstring).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta_num: float
    delta_num_se: float
    delta_den: float
    delta_den_se: float
    delta_cov: float
    treatment_group: str
    n_control: float
    n_treatment: float
    n_enrolled: float
    other_arm_ids: tuple[str, ...]
    site_total_numerator: float
    site_total_denominator: float
    baseline_numerator: float
    baseline_denominator: float
    baseline_ratio: float
    shipped_numerator: float
    shipped_denominator: float
    shipped_ratio: float
    absolute_impact: float
    absolute_impact_se: float
    absolute_impact_lb: float
    absolute_impact_ub: float
    relative_impact: float
    relative_impact_se: float
    relative_impact_lb: float
    relative_impact_ub: float
    alpha: float
    n_clusters: int | None = None
    absolute_dof: float | None = None
    relative_dof: float | None = None


def _checked_arms(
    control: ArmStats, target: ArmStats, other_arms: Sequence[ArmStats]
) -> tuple[ArmStats, ...]:
    """Validate control, target, and co-enrolled arms before any of them
    enters a contrast: metrics must agree, every arm must share one
    study identity, and no ``group_id`` may repeat, or the
    counterfactual baseline would silently corrupt.
    """
    if control.metric != target.metric:
        _raise(
            "estimation.sitewide.control_carries_metric",
            control=control.metric,
            target=target.metric,
        )
    if control.study_id != target.study_id:
        _raise(
            "estimation.sitewide.control_carries_study",
            control=control.study_id,
            target=target.study_id,
        )
    if control.group_id == target.group_id:
        _raise("estimation.sitewide.control_target_are", control=control.group_id)
    seen = {control.group_id, target.group_id}
    for arm in other_arms:
        if arm.metric != target.metric:
            _raise(
                "estimation.sitewide.other_arms_entry_metric_mismatch",
                arm=arm.group_id,
                metric=arm.metric,
                target=target.metric,
            )
        if arm.study_id != target.study_id:
            _raise(
                "estimation.sitewide.other_arms_entry_study_mismatch",
                arm=arm.group_id,
                study_id=arm.study_id,
                target_study_id=target.study_id,
            )
        if arm.group_id in seen:
            _raise("estimation.sitewide.other_arms_repeats", arm=arm.group_id)
        seen.add(arm.group_id)
    return tuple(other_arms)


class SitewideContrast(BaseModel):
    """A sum-metric contrast, normalized to per-unit moments regardless of
    whether it was built from iid units or cluster sums.

    The sole input :func:`sitewide_impact` reads.

    Parameters
    ----------
    metric : str
        The metric this contrast is on.
    control, target : _Arm
        The control arm and the target arm's normalized moments.
    others : tuple[_Arm, ...]
        Every other enrolled non-control arm, normalized the same way.
    n_clusters : int | None
        Total cluster count across control/target/others. ``None`` at iid
        grain.
    dof : float | None
        ``n_clusters - 2``. ``None`` at iid grain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    control: _Arm
    target: _Arm
    others: tuple[_Arm, ...] = ()
    n_clusters: int | None = None
    dof: float | None = None

    @classmethod
    def from_iid(
        cls, control: ArmStats, target: ArmStats, other_arms: Sequence[ArmStats] = ()
    ) -> Self:
        """Normalize an iid (unit-grain) sum-metric contrast.

        Parameters
        ----------
        control, target : ArmStats
            The control arm and the target arm's aggregated moments.
        other_arms : Sequence[ArmStats]
            Every other enrolled non-control arm.

        Returns
        -------
        SitewideContrast

        Raises
        ------
        ValueError
            See :func:`_checked_arms`. Also if ``control``/``target``/any
            ``other_arms`` entry carries a populated ``ref_den``, which is
            only ever set by a clustered row; use :meth:`from_clusters`
            for those instead.
        """
        others = _checked_arms(control, target, other_arms)
        for label, arm in (
            ("control", control),
            ("target", target),
            *((f"other_arms[{i}]", a) for i, a in enumerate(others)),
        ):
            if arm.ref_den is not None:
                _raise(
                    "estimation.sitewide.from_iid.metric_arm_carries_ref_den",
                    target=target.metric,
                    label=label,
                    arm=arm.group_id,
                )

        def _normalized(arm: ArmStats) -> _Arm:
            return _Arm(
                group_id=arm.group_id,
                n_units=float(arm.n),
                mean=arm.mean_y(),
                var_mean=arm.var_y() / arm.n,
            )

        return cls(
            metric=target.metric,
            control=_normalized(control),
            target=_normalized(target),
            others=tuple(_normalized(a) for a in others),
        )

    @classmethod
    def from_clusters(
        cls,
        control: ArmStats,
        target: ArmStats,
        other_arms: Sequence[ArmStats] = (),
        *,
        cluster: str,
    ) -> Self:
        """Normalize a cluster-grain sum-metric contrast.

        Each arm's cluster outcome totals and cluster sizes collapse to
        the same per-unit ``mean``/``var_mean`` shape :meth:`from_iid`
        produces, via the ratio-of-cluster-sums delta method.

        Parameters
        ----------
        control, target : ArmStats
            The control arm and the target arm's clustered moments.
        other_arms : Sequence[ArmStats]
            Every other enrolled non-control arm.
        cluster : str
            The experiment's declared randomization-grain column, named
            on the small-cluster-count guard's message.

        Returns
        -------
        SitewideContrast

        Raises
        ------
        ValueError
            See :func:`_checked_arms`, and
            :func:`~increment.estimation.engine.check_total_clusters`
            below 40 total clusters; every enrolled arm still needs at least
            two clusters.
        """
        others = _checked_arms(control, target, other_arms)
        arms = (control, target, *others)
        n_clusters_total = sum(a.n for a in arms)
        check_total_clusters(target.metric, cluster, n_clusters_total)
        if control.n < 2 or target.n < 2 or any(a.n < 2 for a in others):
            _raise(
                "estimation.sitewide.cluster_arm_needs_two",
                metric=target.metric,
                cluster=cluster,
                arm_counts={a.group_id: a.n for a in arms},
            )
        dof_total = float(n_clusters_total - 2)

        def _normalized(arm: ArmStats) -> _Arm:
            g_bar, _, var_g, _, _ = cluster_outcome_moments(arm)
            m_bar, var_m, cov_gm = arm.mean_x(), arm.var_x(), arm.cov_yx()
            r, se = ratio_abs_diff_se(g_bar, m_bar, var_g, var_m, cov_gm, arm.n)
            return _Arm(
                group_id=arm.group_id,
                n_units=arm.n * arm.mean_den(),
                mean=r,
                var_mean=se * se,
                own_dof=float(arm.n - 1),
            )

        control_arm = _normalized(control)
        target_arm = _normalized(target)
        others_norm = tuple(_normalized(a) for a in others)

        return cls(
            metric=target.metric,
            control=control_arm,
            target=target_arm,
            others=others_norm,
            n_clusters=n_clusters_total,
            dof=dof_total,
        )


class SitewideRatioContrast(CodedModel, BaseModel):
    """A ratio-metric contrast, normalized to per-unit moments regardless
    of whether it was built from iid units or cluster sums.

    The sole input :func:`sitewide_impact_ratio` reads.

    Parameters
    ----------
    metric : str
        The metric this contrast is on.
    control, target : _Arm
        The control arm and the target arm's normalized moments; every
        arm's ``mean_den``/``var_mean_den``/``cov_mean_den`` are
        populated (enforced below).
    others : tuple[_Arm, ...]
        Every other enrolled non-control arm, normalized the same way.
    n_clusters : int | None
        Total cluster count across control/target/others. ``None`` at iid
        grain.
    dof : float | None
        ``n_clusters - 2``. ``None`` at iid grain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    control: _Arm
    target: _Arm
    others: tuple[_Arm, ...] = ()
    n_clusters: int | None = None
    dof: float | None = None

    @model_validator(mode="after")
    def _requires_den_family(self) -> Self:
        for arm in (self.control, self.target, *self.others):
            if arm.mean_den is None or arm.var_mean_den is None or arm.cov_mean_den is None:
                _raise(
                    "estimation.sitewide.sitewide_ratio.metric_arm_missing",
                    metric=self.metric,
                    arm=arm.group_id,
                )
        return self

    @classmethod
    def from_iid(
        cls, control: ArmStats, target: ArmStats, other_arms: Sequence[ArmStats] = ()
    ) -> Self:
        """Normalize an iid (unit-grain) ratio-metric contrast.

        Parameters
        ----------
        control, target : ArmStats
            The control arm and the target arm's aggregated moments.
        other_arms : Sequence[ArmStats]
            Every other enrolled non-control arm.

        Returns
        -------
        SitewideRatioContrast

        Raises
        ------
        ValueError
            See :func:`_checked_arms`. Also if ``control``/``target``/any
            ``other_arms`` entry declares ``x_role='cluster_size'``, which
            is how a clustered ratio row carries cluster size (the den
            family already holds the ratio's own denominator); use
            :meth:`from_clusters` for those instead. Also anything
            :func:`~increment.estimation.variance.ratio_moments` raises
            (missing ratio family, non-positive numerator/denominator
            mean).

        Notes
        -----
        Clustered and iid ratio rows are otherwise wire-identical: a
        clustered row always declares ``x_role='cluster_size'`` (cluster
        size rides the x family since the den family already holds the
        ratio's own denominator), which is what this constructor checks
        to tell them apart.
        """
        others = _checked_arms(control, target, other_arms)
        for label, arm in (
            ("control", control),
            ("target", target),
            *((f"other_arms[{i}]", a) for i, a in enumerate(others)),
        ):
            # Allowlist on the declaration: any non-covariate x slot is cluster grain
            # and would be read here as unit grain, including a role added later.
            if arm.x_role is not None and arm.x_role != X_SLOT_ROLES["x"]:
                # Only a size row belongs in from_clusters, which reads x as mean cluster
                # size; an uptake row has no size to hand over, so it is refused outright.
                remedy = (
                    "build this contrast with SitewideRatioContrast.from_clusters, not from_iid"
                    if arm.x_role == X_SLOT_ROLES["size"]
                    else "sitewide impact is not supported for an encouragement "
                    "uptake row: its x family carries uptake totals, not the cluster "
                    "size from_clusters requires -- rebuild the arm with cluster "
                    "sizes in x, or report this metric through the encouragement path"
                )
                _raise(
                    "estimation.sitewide.sitewide_ratio.metric_arm_declares",
                    target=target.metric,
                    label=label,
                    arm=arm.group_id,
                    x_role=arm.x_role,
                    remedy=remedy,
                )

        def _normalized(arm: ArmStats) -> _Arm:
            num_bar, den_bar, var_num, var_den, cov_numden = ratio_moments(arm)
            return _Arm(
                group_id=arm.group_id,
                n_units=float(arm.n),
                mean=num_bar,
                var_mean=var_num / arm.n,
                mean_den=den_bar,
                var_mean_den=var_den / arm.n,
                cov_mean_den=cov_numden / arm.n,
            )

        return cls(
            metric=target.metric,
            control=_normalized(control),
            target=_normalized(target),
            others=tuple(_normalized(a) for a in others),
        )

    @classmethod
    def from_clusters(
        cls,
        control: ArmStats,
        target: ArmStats,
        other_arms: Sequence[ArmStats] = (),
        *,
        cluster: str,
    ) -> Self:
        """Normalize a cluster-grain ratio-metric contrast.

        Each arm's per-cluster numerator/denominator totals and cluster
        sizes collapse to the same per-unit ``mean``/``mean_den`` shape
        :meth:`from_iid` produces, via the ratio-of-cluster-sums delta
        method.

        Parameters
        ----------
        control, target : ArmStats
            The control arm and the target arm's clustered moments.
        other_arms : Sequence[ArmStats]
            Every other enrolled non-control arm.
        cluster : str
            The experiment's declared randomization-grain column, named
            on the small-cluster-count guard's message.

        Returns
        -------
        SitewideRatioContrast

        Raises
        ------
        ValueError
            If an enrolled arm has fewer than two clusters. A
            :func:`~increment.estimation.engine.check_total_clusters`
            warning is emitted below 40 total clusters.
        """
        others = _checked_arms(control, target, other_arms)
        arms = (control, target, *others)
        n_clusters_total = sum(a.n for a in arms)
        check_total_clusters(target.metric, cluster, n_clusters_total)
        if control.n < 2 or target.n < 2 or any(a.n < 2 for a in others):
            _raise(
                "estimation.sitewide.cluster_arm_needs_two",
                metric=target.metric,
                cluster=cluster,
                arm_counts={a.group_id: a.n for a in arms},
            )
        dof_total = float(n_clusters_total - 2)

        def _normalized(arm: ArmStats) -> _Arm:
            m_bar, var_m = arm.mean_x(), arm.var_x()
            num_bar, var_num, cov_num_m = arm.mean_y(), arm.var_y(), arm.cov_yx()
            den_bar, var_den, cov_den_m = arm.mean_den(), arm.var_den(), arm.cov_xden()
            cov_num_den = arm.cov_yden()
            mean, se_num = ratio_abs_diff_se(num_bar, m_bar, var_num, var_m, cov_num_m, arm.n)
            mean_den, se_den = ratio_abs_diff_se(den_bar, m_bar, var_den, var_m, cov_den_m, arm.n)
            cov_mean_den = ratio_pair_cov(
                mean, mean_den, m_bar, cov_num_den, cov_num_m, cov_den_m, var_m, arm.n
            )
            return _Arm(
                group_id=arm.group_id,
                n_units=arm.n * m_bar,
                mean=mean,
                var_mean=se_num * se_num,
                mean_den=mean_den,
                var_mean_den=se_den * se_den,
                cov_mean_den=cov_mean_den,
                own_dof=float(arm.n - 1),
            )

        control_arm = _normalized(control)
        target_arm = _normalized(target)
        others_norm = tuple(_normalized(a) for a in others)

        return cls(
            metric=target.metric,
            control=control_arm,
            target=target_arm,
            others=others_norm,
            n_clusters=n_clusters_total,
            dof=dof_total,
        )


def _combination_var(
    control: _Arm,
    arms: Sequence[_Arm],
    coef_y: Sequence[float],
    coef_den: Sequence[float] | None = None,
) -> list[float]:
    """Delta-method variance terms of a gradient contracted against arm
    deltas: ``[control_anchored_term, *arm_terms]``, one per arm in
    ``arms`` after the leading control-anchored term.

    Expanded over the independent arm means rather than the correlated
    deltas, threading the shared-control covariance
    ``Cov(delta_i, delta_j) = control.var_mean`` without assembling a
    matrix (see "Multi-arm experiments" in the module docstring). Returns
    terms rather than their sum so callers can both sum them for the
    total variance and pair them with each term's own dof for a
    Satterthwaite reduction. *coef_den* is ``None`` for a sum-type metric.
    """
    sum_y = math.fsum(coef_y)
    control_term = sum_y * sum_y * control.var_mean
    arm_terms = [a * a * arm.var_mean for arm, a in zip(arms, coef_y, strict=True)]
    if coef_den is not None:
        sum_den = math.fsum(coef_den)
        control_var_den = control.var_mean_den
        control_cov_den = control.cov_mean_den
        assert control_var_den is not None and control_cov_den is not None
        control_term += (
            sum_den * sum_den * control_var_den + 2.0 * sum_y * sum_den * control_cov_den
        )
        for i, (arm, a, b) in enumerate(zip(arms, coef_y, coef_den, strict=True)):
            arm_var_den = arm.var_mean_den
            arm_cov_den = arm.cov_mean_den
            assert arm_var_den is not None and arm_cov_den is not None
            arm_terms[i] += b * b * arm_var_den + 2.0 * a * b * arm_cov_den
    return [control_term, *arm_terms]


def _validate_alpha(alpha: float) -> None:
    """Reject a two-sided significance level outside its probability domain."""
    if not 0.0 < alpha < 1.0:
        _raise("estimation.sitewide.alpha", alpha=alpha)


def _validate_site_total(name: str, value: float) -> None:
    """Reject a non-finite site-window total before any arithmetic uses it.

    A site total may legitimately be zero or negative (a signed metric
    like profit), so only finiteness is required here; the derived
    baseline/shipped totals downstream carry their own positivity checks.
    """
    if not math.isfinite(value):
        _raise("estimation.armstats.arm_stats.cross_field_finite", name=name, value=value)


def _critical_value(alpha: float, dof: float | None) -> float:
    """Two-sided Wald critical value: ``t_{dof}`` under a cluster-robust
    reference, Normal ``z`` otherwise.
    """
    if dof is not None:
        return two_sided_critical_value(student_t_isf, alpha, dof, what="sitewide impact interval")
    return two_sided_critical_value(_norm.isf, alpha, what="sitewide impact interval")


def _satterthwaite_dof(components: Sequence[float], arms: Sequence[_Arm], control: _Arm) -> float:
    """Welch-Satterthwaite dof over :func:`_combination_var`'s per-term
    variance components, each paired with the degrees of the ONE arm whose
    clusters estimated it (``_Arm.own_dof``, ``K_i - 1``): *control*'s own
    for the shared control-anchored term, each arm's own for its term.

    Neither the contrast's pooled ``n_clusters - 2`` nor an arm's pairwise
    ``K_i + K_C - 2`` belongs here: both credit a term with cluster degrees
    its variance estimate never touched, and can push the reduction above
    the bound ``sum_i(K_i - 1)`` a Satterthwaite dof can never exceed.
    """
    assert control.own_dof is not None
    dofs: list[float] = [control.own_dof]
    for arm in arms:
        assert arm.own_dof is not None
        dofs.append(arm.own_dof)
    sum_v = math.fsum(components)
    denom = math.fsum(v * v / d for v, d in zip(components, dofs, strict=True))
    return sum_v * sum_v / denom if denom > 0 else control.own_dof


def sitewide_impact(
    contrast: SitewideContrast,
    *,
    site_total_volume: float,
    alpha: float = 0.05,
) -> SitewideImpact:
    """Ship-to-all site impact of a sum-type metric's lift.

    Answers: if *contrast*'s target arm had shipped to every enrolled
    unit and no other treatment arm had ever been exposed, how much
    higher would the site-window total be than under no treatment at
    all? See "Multi-arm experiments" in the module docstring for why
    that all-control counterfactual, not the partially-exposed status
    quo, is the baseline every arm is scored against.

    When *contrast* was built from clusters, a ``UserWarning`` is
    emitted: the baseline assumes enrolled units exhaust the treated
    population, which cluster randomization does not guarantee, so
    relative impact is then a lower bound; absolute impact is unaffected.

    Parameters
    ----------
    contrast : SitewideContrast
        The control/target/other-arms moments, already normalized to one
        grain.
    site_total_volume : float
        Total metric volume over the whole site (not just enrolled units)
        during the experiment's enrollment window, computed independently
        by the query layer from the metric's raw event tables.
    alpha : float
        Two-sided significance level for both intervals. Default 0.05.

    Returns
    -------
    SitewideImpact

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly between 0 and 1, if
        ``site_total_volume`` is not finite, or if the counterfactual
        baseline volume (``V0``) is not strictly positive, which makes
        relative impact undefined.
    """
    _validate_alpha(alpha)
    _validate_site_total("site_total_volume", site_total_volume)
    control = contrast.control
    target = contrast.target
    others = contrast.others
    arms = (target, *others)
    if contrast.n_clusters is not None:
        _warn_cluster_baseline_assumption()

    delta = target.mean - control.mean
    delta_se = math.sqrt(target.var_mean + control.var_mean)

    n_exp = control.n_units + sum(arm.n_units for arm in arms)
    # Every enrolled arm's lift is in the observed total, so every one of
    # them leaves the counterfactual baseline.
    other_contribution = math.fsum(arm.n_units * (arm.mean - control.mean) for arm in others)
    baseline_volume = site_total_volume - delta * target.n_units - other_contribution
    if baseline_volume <= 0:
        _raise(
            "estimation.sitewide.counterfactual_baseline_volume",
            baseline_volume=baseline_volume,
            site_total_volume=site_total_volume,
            delta=delta,
            n_units=target.n_units,
            other_arms=[arm.group_id for arm in others],
            other_contribution=other_contribution,
        )

    # Linear in the target delta: use Welch-Satterthwaite over the two
    # components already summed by `delta_se`, each retaining its arm's
    # Bessel-corrected degrees of freedom rather than a pooled reference.
    absolute_dof: float | None = None
    if contrast.n_clusters is not None:
        assert control.own_dof is not None and target.own_dof is not None
        absolute_dof = welch_satterthwaite_df(
            target.var_mean, target.own_dof, control.var_mean, control.own_dof
        )
    crit_abs = _critical_value(alpha, absolute_dof)
    absolute_impact = delta * n_exp
    absolute_impact_se = delta_se * n_exp
    absolute_impact_lb = absolute_impact - crit_abs * absolute_impact_se
    absolute_impact_ub = absolute_impact + crit_abs * absolute_impact_se

    relative_impact = absolute_impact / baseline_volume
    # Quotient-rule delta method (module docstring); each other arm
    # contributes one partial through the shared baseline.
    volume_net_of_others = site_total_volume - other_contribution
    grad = [n_exp * volume_net_of_others / baseline_volume**2]
    grad += [absolute_impact * arm.n_units / baseline_volume**2 for arm in others]
    components = _combination_var(control, arms, grad)
    relative_impact_var = max(math.fsum(components), 0.0)
    relative_impact_se = math.sqrt(relative_impact_var)
    relative_dof: float | None = None
    if contrast.n_clusters is not None:
        assert contrast.dof is not None
        relative_dof = _satterthwaite_dof(components, arms, control)
    crit_rel = _critical_value(alpha, relative_dof)
    relative_impact_lb = relative_impact - crit_rel * relative_impact_se
    relative_impact_ub = relative_impact + crit_rel * relative_impact_se

    return SitewideImpact(
        delta=delta,
        delta_se=delta_se,
        treatment_group=target.group_id,
        n_control=control.n_units,
        n_treatment=target.n_units,
        n_enrolled=n_exp,
        other_arm_ids=tuple(arm.group_id for arm in others),
        site_total_volume=site_total_volume,
        baseline_volume=baseline_volume,
        absolute_impact=absolute_impact,
        absolute_impact_se=absolute_impact_se,
        absolute_impact_lb=absolute_impact_lb,
        absolute_impact_ub=absolute_impact_ub,
        relative_impact=relative_impact,
        relative_impact_se=relative_impact_se,
        relative_impact_lb=relative_impact_lb,
        relative_impact_ub=relative_impact_ub,
        alpha=alpha,
        n_clusters=contrast.n_clusters,
        absolute_dof=absolute_dof,
        relative_dof=relative_dof,
    )


def sitewide_impact_ratio(  # noqa: PLR0915
    contrast: SitewideRatioContrast,
    *,
    site_total_numerator: float,
    site_total_denominator: float,
    alpha: float = 0.05,
) -> SitewideRatioImpact:
    """Ship-to-all site impact of a ratio metric's lift.

    Answers the same question :func:`sitewide_impact` does for a metric
    whose per-unit value is a ratio.

    As with :func:`sitewide_impact`, a cluster-built *contrast* emits a
    ``UserWarning`` that the baseline assumes enrolled units exhaust the
    treated population.

    Parameters
    ----------
    contrast : SitewideRatioContrast
        The control/target/other-arms moments, already normalized to one
        grain.
    site_total_numerator, site_total_denominator : float
        Total numerator/denominator volume over the whole site (not just
        enrolled units) during the experiment's enrollment window,
        computed independently by the query layer.
    alpha : float
        Two-sided significance level for the interval. Default 0.05.

    Returns
    -------
    SitewideRatioImpact

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly between 0 and 1, if
        ``site_total_numerator``/``site_total_denominator`` are not
        finite, or if the counterfactual baseline denominator (``D0``)
        or the ship-to-all denominator (``D1``) is not strictly
        positive, since both are divisors in the impact itself. Also if
        the counterfactual baseline ratio (``N0/D0``) is not strictly
        positive, which would make relative impact a nonsense sign or a
        division by zero.
    """
    _validate_alpha(alpha)
    _validate_site_total("site_total_numerator", site_total_numerator)
    _validate_site_total("site_total_denominator", site_total_denominator)
    control = contrast.control
    target = contrast.target
    others = contrast.others
    arms = (target, *others)
    assert target.mean_den is not None and control.mean_den is not None
    assert target.var_mean_den is not None and control.var_mean_den is not None
    assert target.cov_mean_den is not None and control.cov_mean_den is not None
    if contrast.n_clusters is not None:
        _warn_cluster_baseline_assumption()

    delta_num = target.mean - control.mean
    delta_den = target.mean_den - control.mean_den
    delta_num_se = math.sqrt(target.var_mean + control.var_mean)
    delta_den_se = math.sqrt(target.var_mean_den + control.var_mean_den)
    delta_cov = target.cov_mean_den + control.cov_mean_den

    n_exp = control.n_units + sum(arm.n_units for arm in arms)
    # Both site totals carry every enrolled arm's lift, so both baselines
    # net out every arm, not just the target one.
    num_contribution = math.fsum(arm.n_units * (arm.mean - control.mean) for arm in arms)

    def _den(arm: _Arm) -> float:
        assert arm.mean_den is not None
        return arm.mean_den

    den_contribution = math.fsum(arm.n_units * (_den(arm) - control.mean_den) for arm in arms)
    baseline_numerator = site_total_numerator - num_contribution
    baseline_denominator = site_total_denominator - den_contribution
    other_ids = [arm.group_id for arm in others]
    if baseline_denominator <= 0:
        _raise(
            "estimation.sitewide.counterfactual_baseline_denominator",
            baseline_denominator=baseline_denominator,
            site_total_denominator=site_total_denominator,
            delta_den=delta_den,
            n_units=target.n_units,
            other_arms=other_ids,
            den_contribution=den_contribution,
        )

    shipped_numerator = baseline_numerator + n_exp * delta_num
    shipped_denominator = baseline_denominator + n_exp * delta_den
    if shipped_denominator <= 0:
        _raise(
            "estimation.sitewide.ship_all_denominator",
            shipped_denominator=shipped_denominator,
            baseline_denominator=baseline_denominator,
            delta_den=delta_den,
            n_exp=n_exp,
        )

    if baseline_numerator <= 0:
        _raise(
            "estimation.sitewide.counterfactual_baseline_ratio",
            baseline_numerator=baseline_numerator,
            baseline_denominator=baseline_denominator,
            site_total_numerator=site_total_numerator,
            delta_num=delta_num,
            n_units=target.n_units,
            other_arms=other_ids,
            num_contribution=num_contribution,
        )

    baseline_ratio = baseline_numerator / baseline_denominator
    shipped_ratio = shipped_numerator / shipped_denominator
    absolute_impact = shipped_ratio - baseline_ratio

    # Gradient (module docstring, "The ratio math"). n_switched is every
    # unit that would move into the target arm.
    n_switched = n_exp - target.n_units
    g_num = [n_switched / shipped_denominator + target.n_units / baseline_denominator]
    g_num += [
        arm.n_units * (1.0 / baseline_denominator - 1.0 / shipped_denominator) for arm in others
    ]
    g_den = [
        -shipped_numerator * n_switched / shipped_denominator**2
        - baseline_numerator * target.n_units / baseline_denominator**2
    ]
    g_den += [
        arm.n_units
        * (
            shipped_numerator / shipped_denominator**2
            - baseline_numerator / baseline_denominator**2
        )
        for arm in others
    ]
    abs_components = _combination_var(control, arms, g_num, g_den)
    absolute_impact_var = max(math.fsum(abs_components), 0.0)
    absolute_impact_se = math.sqrt(absolute_impact_var)
    absolute_dof: float | None = None
    if contrast.n_clusters is not None:
        assert contrast.dof is not None
        absolute_dof = _satterthwaite_dof(abs_components, arms, control)
    crit_abs = _critical_value(alpha, absolute_dof)
    absolute_impact_lb = absolute_impact - crit_abs * absolute_impact_se
    absolute_impact_ub = absolute_impact + crit_abs * absolute_impact_se

    relative_impact = absolute_impact / baseline_ratio
    # Product-rule delta method on rel = (N1/N0)*(D0/D1) - 1; each factor
    # depends on the numerator or denominator lifts only, never both.
    den_ratio = baseline_denominator / shipped_denominator
    num_ratio = shipped_numerator / baseline_numerator
    rel_g_num = [
        den_ratio
        * (n_switched * baseline_numerator + target.n_units * shipped_numerator)
        / baseline_numerator**2
    ]
    rel_g_num += [
        den_ratio * arm.n_units * (shipped_numerator - baseline_numerator) / baseline_numerator**2
        for arm in others
    ]
    rel_g_den = [
        -num_ratio
        * (target.n_units * shipped_denominator + n_switched * baseline_denominator)
        / shipped_denominator**2
    ]
    rel_g_den += [
        num_ratio
        * arm.n_units
        * (baseline_denominator - shipped_denominator)
        / shipped_denominator**2
        for arm in others
    ]
    rel_components = _combination_var(control, arms, rel_g_num, rel_g_den)
    relative_impact_var = max(math.fsum(rel_components), 0.0)
    relative_impact_se = math.sqrt(relative_impact_var)
    relative_dof: float | None = None
    if contrast.n_clusters is not None:
        assert contrast.dof is not None
        relative_dof = _satterthwaite_dof(rel_components, arms, control)
    crit_rel = _critical_value(alpha, relative_dof)
    relative_impact_lb = relative_impact - crit_rel * relative_impact_se
    relative_impact_ub = relative_impact + crit_rel * relative_impact_se

    return SitewideRatioImpact(
        delta_num=delta_num,
        delta_num_se=delta_num_se,
        delta_den=delta_den,
        delta_den_se=delta_den_se,
        delta_cov=delta_cov,
        treatment_group=target.group_id,
        n_control=control.n_units,
        n_treatment=target.n_units,
        n_enrolled=n_exp,
        other_arm_ids=tuple(other_ids),
        site_total_numerator=site_total_numerator,
        site_total_denominator=site_total_denominator,
        baseline_numerator=baseline_numerator,
        baseline_denominator=baseline_denominator,
        baseline_ratio=baseline_ratio,
        shipped_numerator=shipped_numerator,
        shipped_denominator=shipped_denominator,
        shipped_ratio=shipped_ratio,
        absolute_impact=absolute_impact,
        absolute_impact_se=absolute_impact_se,
        absolute_impact_lb=absolute_impact_lb,
        absolute_impact_ub=absolute_impact_ub,
        relative_impact=relative_impact,
        relative_impact_se=relative_impact_se,
        relative_impact_lb=relative_impact_lb,
        relative_impact_ub=relative_impact_ub,
        alpha=alpha,
        n_clusters=contrast.n_clusters,
        absolute_dof=absolute_dof,
        relative_dof=relative_dof,
    )
