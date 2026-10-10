"""Plan two-segment lift contrasts with the baseline-only variance model.

For each segment, the planning variance is ``v / m^2 * (1/n_T + 1/n_C)``
using that segment's baseline in both arms. This is a documented approximation
that is distinct from the arm solvers' variance evaluated at each alternative.
Quantile metrics are refused because quantile readouts have no segments.
"""

from __future__ import annotations

import math
from typing import cast

from scipy.stats import norm as _norm

from increment._finite_sample_refusals import refuse_finite_sample_metric_type
from increment.compatibility import Unsupported, refuse_unsupported
from increment.decision import FixedInference
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals, refuse
from increment.estimation._readout_refusals import READOUT_REFUSALS
from increment.estimation.arm_contract import (
    ArmPlanningProcedure,
    RelativeDecisionPolicy,
    arm_planning_support,
)
from increment.estimation.conversion_route import (
    finite_sample_blocker,
    refuse_finite_sample_unavailable,
)
from increment.power._shared import (
    POWER_SOLVERS_RELATIVE,
    derive_axes_from_baseline,
)
from increment.power._shared import (
    cluster_counts as _cluster_counts,
)
from increment.power._shared import (
    compute_arms as _compute_arms,
)
from increment.power._shared import (
    mde_theta as _mde_theta,
)
from increment.power._shared import (
    power_at as _power_at,
)
from increment.power._shared import (
    validate_relative_lift as _validate_relative_lift,
)
from increment.power.core import Baseline, PowerDesign, PowerResult, QuantileBaseline

_PAIRWISE_N_PER_ARM_TOO_SMALL = RefusalSpec(
    "power.segment_pairwise_achieved_n_per_arm_too_small",
    InvalidRequestError,
    template=(
        "n_per_arm={n_per_arm} is too small to split segments A (q={q_a}) and B "
        "(q={q_b}) into 2-arm designs ({exc}); increase n_per_arm or the "
        "smaller segment's share"
    ),
)

_PAIRWISE_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.segment_share_n": "segment share q={q} of n_total={n_total} implies only {n_total_seg:.3g} units, too few for a 2-arm design at allocation={allocation} (need >= {min_n_total_seg:.3g} total at this allocation); increase n_total or the segment's share",
        "power.q_a": "q_a must be in (0, 1)",
        "power.q_b": "q_b must be in (0, 1)",
        "power.q_a_q": "q_a + q_b must be <= 1, got {q_a_plus_q_b}",
        "power.segment_pairwise_solvers": RefusalSpec(
            "power.segment_pairwise_solvers",
            InvalidRequestError,
            lambda *, unsupported: (
                "segment-pairwise solvers do not support non-default Baseline fields: "
                + ", ".join(unsupported)
            ),
        ),
        "power.segment_clustered_baseline": "segment {segment}'s clustered baseline requires at least two clusters per arm (got {k_total} total, n_t={n_t}, n_c={n_c}); increase n_total or the segment's share",
        "power.power_solvers_relative": POWER_SOLVERS_RELATIVE,
        "power.supports_fixed_horizon": "{caller} supports fixed-horizon ArmPlanningProcedure only",
        "power.r_a_below": "r_a ({r_a}) is below r_b ({r_b}) (theta={theta:.6g}) but alternative='greater' -- no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b",
        "power.r_a_above": "r_a ({r_a}) is above r_b ({r_b}) (theta={theta:.6g}) but alternative='less' -- no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b",
        "power.r_a_r": "r_a ({r_a}) and r_b ({r_b}) give the same relative lift (theta=0); no finite sample size can distinguish segment A from segment B in this design",
        "power.solved_too_small": "the solved-for N ({n_total}) is too small to split segments A (q={q_a}) and B (q={q_b}) into 2-arm designs ({exc}); this can happen when the contrast between r_a and r_b is large relative to the smaller segment's share, needing more of the solved-for N per segment than a 2-arm split allows -- narrow the contrast, raise the smaller segment's share, or accept that an easy-to-detect contrast needs a manually-chosen larger N",
        "power.segment_pairwise_required": "segment_pairwise_required_sample_size does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.segment_pairwise_achieved": "segment_pairwise_achieved_power does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.segment_pairwise_minimum": "segment_pairwise_minimum_detectable_effect does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.segment_pairwise_achieved_n_per_arm_too_small": _PAIRWISE_N_PER_ARM_TOO_SMALL,
    },
)
_raise = raiser(_PAIRWISE_REFUSALS)


def _se_sq(n_T: float, n_C: float, baseline: Baseline) -> float:
    """Baseline-only log-ratio variance ``v / m^2 * (1/n_T + 1/n_C)``.

    Retained for the segment-pairwise solvers, whose documented model
    evaluates every arm at its segment's baseline. The arm trio uses
    ``_arm_log_se_sq`` instead.
    """
    return baseline.effective_var / (baseline.mean**2) * (1.0 / n_T + 1.0 / n_C)


def _mde_relative(
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Minimum detectable relative effect at a fixed variance ``se2``
    (segment-pairwise model)."""
    return math.expm1(_mde_theta(se2, design, procedure, dof=dof))


# Pairwise segment-difference solvers - detect a difference between two
# segments' lifts; see segment_pairwise_required_sample_size for the formula.


def _segment_arm_sizes(q: float, n_total: float, design: PowerDesign) -> tuple[int, int]:
    """Split a segment share into treatment/control arms.

    Every derived arm holds at least two units, so the allocation-aware
    segment threshold is ``2 / min(allocation, 1 - allocation)``; below it,
    flooring the arms would fabricate units not present in the segment.
    """
    n_total_seg = q * n_total
    min_alloc = min(design.allocation, 1.0 - design.allocation)
    min_n_total_seg = 2 / min_alloc
    if n_total_seg < min_n_total_seg:
        _raise(
            "power.segment_share_n",
            q=q,
            n_total=n_total,
            n_total_seg=n_total_seg,
            allocation=design.allocation,
            min_n_total_seg=min_n_total_seg,
        )
    n_t = max(2, math.ceil(n_total_seg * design.allocation))
    n_c = max(2, math.ceil(n_total_seg * (1.0 - design.allocation)))
    return n_t, n_c


def _pairwise_theta(r_a: float, r_b: float) -> float:
    _validate_relative_lift(r_a)
    _validate_relative_lift(r_b)
    return math.log1p(r_a) - math.log1p(r_b)


def _validate_segment_shares(q_a: float, q_b: float) -> None:
    if not 0 < q_a < 1:
        _raise("power.q_a")
    if not 0 < q_b < 1:
        _raise("power.q_b")
    if q_a + q_b > 1.0 + 1e-9:
        _raise("power.q_a_q", q_a_plus_q_b=q_a + q_b)


def _validate_pairwise_baselines(baseline_a: Baseline, baseline_b: Baseline) -> None:
    unsupported = [
        field
        for field in ("compliance", "trigger_rate")
        if any(getattr(baseline, field) != 1.0 for baseline in (baseline_a, baseline_b))
    ]
    if unsupported:
        _raise("power.segment_pairwise_solvers", unsupported=unsupported)


def _prepare_pairwise_baseline(
    procedure: ArmPlanningProcedure, baseline: Baseline
) -> tuple[ArmPlanningProcedure, Baseline]:
    procedure = derive_axes_from_baseline(procedure, baseline)
    if procedure.decision_method.variance_reduction == "none" and baseline.cuped_rho != 0.0:
        baseline = baseline.model_copy(update={"cuped_rho": 0.0})
    support = arm_planning_support(procedure, baseline=baseline)
    if isinstance(support, Unsupported):
        refuse_unsupported(support)
    _require_pairwise_finite_sample_plannable(procedure)
    return procedure, baseline


def _require_pairwise_finite_sample_plannable(procedure: ArmPlanningProcedure) -> None:
    runtime_binomial = (
        procedure.metric.metric_type in ("conversion", "retention")
        and procedure.dependence == "iid"
        and procedure.decision_method.variance_reduction == "none"
        and isinstance(procedure.inference, FixedInference)
        and not procedure.prior_present
        and procedure.analysis.variance_adjustment == "none"
        and procedure.metric.winsorization == "none"
    )
    if procedure.decision_method.conversion_inference != "finite_sample" or runtime_binomial:
        return
    if procedure.metric.metric_type not in ("conversion", "retention"):
        refuse_finite_sample_metric_type(procedure.metric.metric_type)
    reason = finite_sample_blocker(
        cluster="the plan's cluster" if procedure.dependence == "cluster" else None,
        prior_present=procedure.prior_present,
        sequential=not isinstance(procedure.inference, FixedInference),
    )
    refuse_finite_sample_unavailable(
        procedure.metric.metric_type,
        reason
        or "the plan adjusts the outcome (variance adjustment, factor absorption or winsorization)",
    )


def _validate_pairwise_cluster_floor(
    n_t: int, n_c: int, baseline: Baseline, *, segment: str
) -> None:
    """Require the analyzer's structural minimum of two clusters per arm."""
    if baseline.avg_cluster_size <= 1.0:
        return
    k_t, k_total = _cluster_counts(n_t, n_c, baseline)
    assert k_t is not None and k_total is not None
    k_c = k_total - k_t
    if k_t < 2 or k_c < 2:
        _raise(
            "power.segment_clustered_baseline",
            segment=segment,
            k_total=k_total,
            n_t=n_t,
            n_c=n_c,
        )


def _pairwise_se_sq(
    n_total: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    baseline_b: Baseline,
    design: PowerDesign,
) -> float:
    """Variance of the segment-A-minus-segment-B log-lift contrast at total N."""
    n_a_t, n_a_c = _segment_arm_sizes(q_a, n_total, design)
    n_b_t, n_b_c = _segment_arm_sizes(q_b, n_total, design)
    _validate_pairwise_cluster_floor(n_a_t, n_a_c, baseline_a, segment="A")
    _validate_pairwise_cluster_floor(n_b_t, n_b_c, baseline_b, segment="B")
    return _se_sq(n_a_t, n_a_c, baseline_a) + _se_sq(n_b_t, n_b_c, baseline_b)


def _prepare_pairwise(
    procedure: ArmPlanningProcedure,
    baseline_a: Baseline,
    baseline_b: Baseline | None,
    design: PowerDesign | None,
    *,
    q_a: float,
    q_b: float,
    caller: str,
) -> tuple[ArmPlanningProcedure, Baseline, Baseline, PowerDesign, RelativeDecisionPolicy]:
    """Shared preamble for the three segment-pairwise solvers: baseline
    prep, design defaulting, and the fixed-horizon/segment-share guards.

    A quantile metric is refused with the readout's own breakout code: its
    segments have no readout to plan for, and a ``QuantileBaseline``
    carries no per-unit variance the segment model could scale."""
    quantile_baseline = next(
        (b for b in (baseline_a, baseline_b) if isinstance(b, QuantileBaseline)), None
    )
    if quantile_baseline is not None or (
        isinstance(procedure, ArmPlanningProcedure) and procedure.metric.metric_type == "quantile"
    ):
        refuse(
            READOUT_REFUSALS["readout.metric.quantile_breakout"],
            metric=None if quantile_baseline is None else quantile_baseline.metric_name,
            solver=caller,
            route=(
                "plan the whole-population quantile lift with required_sample_size, "
                "achieved_power or minimum_detectable_effect"
            ),
        )
    procedure = ArmPlanningProcedure.model_validate(procedure)
    if not isinstance(procedure.decision, RelativeDecisionPolicy):
        _raise("power.power_solvers_relative")
    baseline_a = Baseline.model_validate(baseline_a)
    procedure, baseline_a = _prepare_pairwise_baseline(procedure, baseline_a)
    if baseline_b is None:
        baseline_b = baseline_a
    else:
        baseline_b = Baseline.model_validate(baseline_b)
        procedure, baseline_b = _prepare_pairwise_baseline(procedure, baseline_b)
    _validate_pairwise_baselines(baseline_a, baseline_b)

    if design is None:
        design = PowerDesign()
    else:
        design = PowerDesign.model_validate(design)
    if not isinstance(procedure.inference, FixedInference):
        _raise("power.supports_fixed_horizon", caller=caller)
    _validate_segment_shares(q_a, q_b)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    return procedure, baseline_a, baseline_b, design, decision


def _reraise_cluster_floor(exc: ValueError) -> None:
    """The cluster-floor refusal keeps its own code; other split failures
    become the calling solver's own refusal."""
    if isinstance(exc, CodedError) and exc.code == "power.segment_clustered_baseline":
        raise exc


def segment_pairwise_required_sample_size(
    r_a: float,
    r_b: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Experiment-wide N needed to detect segment A's lift differing from segment B's.

    ``r_a``/``r_b`` are the two segments' relative lifts; ``q_a``/``q_b``
    are their shares of the *whole* experiment (a 10% segment out of ten
    still has ``q=0.10`` regardless of how many other segments exist).
    Cost is ``n_total = n_ATE(delta) * (1/q_a + 1/q_b)``, where
    ``delta = log1p(r_a) - log1p(r_b)`` is the log-scale contrast (not
    ``log1p(r_a - r_b)``: that errs -27.7% at (0.50, 0.20) and +19.4% at
    (0.30, -0.10)), and ``n_ATE(delta)`` is the total N a standard
    50/50-allocation experiment would need to detect ``delta`` as a plain
    ATE. ``1/q_a + 1/q_b`` reduces to the simpler
    ``n_ATE(delta)/(q(1-q))`` form only when ``q_a + q_b = 1`` (a
    2-segment breakout); with more segments the two forms diverge and the
    simpler one understates N.

    ``n_per_arm``/``n_total`` are experiment-wide (summed across every
    segment, not just A and B). ``effective_var`` reflects segment A's
    baseline only. ``baseline_b`` defaults to ``baseline_a``;
    ``design.allocation`` governs the treatment/control split within
    each segment. Cluster design effects flow in via each baseline's
    ``effective_var``, but ``PowerResult.n_clusters_*`` stays ``None``:
    no single cluster count is meaningful across two possibly-different
    baselines. Fixed-horizon only; sequential planning for segment
    contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``r_a`` and ``r_b`` give the same relative lift (theta=0), the
        same refusal ``required_sample_size`` makes for a lift exactly
        at its null boundary. Also raised if the solved-for N is too
        small for either segment's 2-arm split (see ``_segment_arm_sizes``).
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_required_sample_size",
    )
    theta = _pairwise_theta(r_a, r_b)
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_required", null_lift=decision.null_lift)
    alternative = decision.alternative
    if alternative == "greater" and theta < 0.0:
        _raise("power.r_a_below", r_a=r_a, r_b=r_b, theta=theta)
    if alternative == "less" and theta > 0.0:
        _raise("power.r_a_above", r_a=r_a, r_b=r_b, theta=theta)

    k = 1.0 / design.allocation + 1.0 / (1.0 - design.allocation)
    term_a = baseline_a.effective_var / (baseline_a.mean**2 * q_a)
    term_b = baseline_b.effective_var / (baseline_b.mean**2 * q_b)
    z_sum = float(_norm.isf(procedure.compiled_tail_alpha)) + float(_norm.ppf(design.power))

    if abs(theta) < 1e-15:
        _raise("power.r_a_r", r_a=r_a, r_b=r_b)

    n_total_float = k * z_sum**2 * (term_a + term_b) / theta**2
    n_t = max(2, math.ceil(n_total_float * design.allocation))
    n_t, n_c = _compute_arms(n_t, design)

    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise("power.solved_too_small", n_total=n_t + n_c, q_a=q_a, q_b=q_b, exc=str(exc))
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=_mde_relative(se2, design, procedure),
        effective_var=baseline_a.effective_var,
    )


def segment_pairwise_achieved_power(
    n_per_arm: int,
    r_a: float,
    r_b: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Achieved power for detecting segment A's lift differing from segment B's
    at an experiment-wide treatment-arm size of ``n_per_arm``.

    ``PowerResult.effective_var`` reflects segment A's baseline only; it
    does not summarize ``baseline_b``. Fixed-horizon only; sequential
    planning for segment contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``n_per_arm`` implies too few units in either segment's share
        for a 2-arm split (see ``_segment_arm_sizes``); increase
        ``n_per_arm`` or the smaller segment's share.
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_achieved_power",
    )
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_achieved", null_lift=decision.null_lift)
    theta = _pairwise_theta(r_a, r_b)

    n_t, n_c = _compute_arms(n_per_arm, design)
    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise(
            "power.segment_pairwise_achieved_n_per_arm_too_small",
            n_per_arm=n_per_arm,
            q_a=q_a,
            q_b=q_b,
            exc=str(exc),
        )
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=_mde_relative(se2, design, procedure),
        effective_var=baseline_a.effective_var,
    )


def segment_pairwise_minimum_detectable_effect(
    n_per_arm: int,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Smallest segment-A-vs-segment-B difference detectable at ``n_per_arm``.

    ``mde_relative`` is ``exp(delta) - 1`` for the smallest detectable
    log-scale contrast ``delta = log(1+r_A) - log(1+r_B)``: the smallest
    detectable ratio ``(1+r_A)/(1+r_B) - 1``, not a lift against a
    single baseline mean.

    ``PowerResult.effective_var`` reflects segment A's baseline only; it
    does not summarize ``baseline_b``. Fixed-horizon only; sequential
    planning for segment contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``n_per_arm`` implies too few units in either segment's share
        for a 2-arm split (see ``_segment_arm_sizes``); increase
        ``n_per_arm`` or the smaller segment's share.
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_minimum_detectable_effect",
    )
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_minimum", null_lift=decision.null_lift)

    n_t, n_c = _compute_arms(n_per_arm, design)
    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise(
            "power.segment_pairwise_achieved_n_per_arm_too_small",
            n_per_arm=n_per_arm,
            q_a=q_a,
            q_b=q_b,
            exc=str(exc),
        )
    mde_theta = _mde_theta(se2, design, procedure)
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(mde_theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=math.expm1(mde_theta),
        effective_var=baseline_a.effective_var,
    )
