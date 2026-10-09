"""Shared fixed-variance solver primitives used by power planners."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from scipy.stats import norm as _norm

from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.power._noncentral_t import _scalar_power_from_nc
from increment.power._validation import _require_relative_domain
from increment.semantics.models import MethodSpec

if TYPE_CHECKING:
    from increment.compatibility import PowerDesign
    from increment.estimation.arm_contract import ArmPlanningProcedure, RelativeDecisionPolicy

POWER_SOLVERS_RELATIVE = RefusalSpec(
    "power.power_solvers_relative",
    InvalidRequestError,
    template="power solvers require a relative ArmPlanningProcedure decision",
)


_POWER_UNRESOLVED = RefusalSpec(
    "power.noncentral_t_unresolved",
    InvalidRequestError,
    template=(
        "noncentral-t power at noncentrality {nc!r} with {dof!r} degrees of freedom "
        "and tail allocation {tail_alpha!r} is unresolved: its tail integral did not "
        "resolve within the quadrature's panel limits"
    ),
)


def cluster_counts(n_t: int, n_c: int, baseline: Any) -> tuple[int | None, int | None]:
    """Required randomized clusters from assigned units and assigned mean size."""
    mean_size = baseline.avg_cluster_size
    if mean_size <= 1.0:
        return None, None
    k_t = math.ceil(n_t / mean_size)
    return k_t, k_t + math.ceil(n_c / mean_size)


def compute_arms(n_t: int, design: PowerDesign, *, minimum_per_arm: int = 2) -> tuple[int, int]:
    """Return treatment and control arm sizes at the requested allocation."""
    if n_t < minimum_per_arm:
        n_t = minimum_per_arm
    n_c = max(minimum_per_arm, math.ceil(n_t * (1.0 - design.allocation) / design.allocation))
    return n_t, n_c


def power_from_nc(nc: float, procedure: ArmPlanningProcedure, *, dof: float | None = None) -> float:
    """Power at noncentrality signed toward the alternative."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    tail_alpha = procedure.compiled_tail_alpha
    power = _scalar_power_from_nc(
        nc, alternative=decision.alternative, tail_alpha=tail_alpha, dof=dof
    )
    if math.isnan(power):
        refuse(_POWER_UNRESOLVED, nc=nc, dof=dof, tail_alpha=tail_alpha)
    return power


def power_at(
    theta: float,
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Power under the normal/noncentral-t reference at fixed variance."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    theta0 = math.log1p(decision.null_lift)
    nc = (theta - theta0) / math.sqrt(se2)
    return power_from_nc(nc, procedure, dof=dof)


def mde_theta(
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Minimum detectable log-scale effect at a fixed variance."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    se = math.sqrt(se2)
    theta0 = math.log1p(decision.null_lift)
    null_power = power_at(theta0, se2, design, procedure, dof=dof)
    if design.power <= null_power:
        return 0.0
    z_alpha = float(_norm.isf(procedure.compiled_tail_alpha))
    z_power = float(_norm.ppf(design.power))
    alternative = decision.alternative
    direction = 1.0 if alternative in ("two-sided", "greater") else -1.0
    if dof is None and alternative != "two-sided":
        return direction * se * (z_alpha + z_power)

    from scipy.optimize import brentq

    target = design.power

    def gap(nc_abs: float) -> float:
        return power_from_nc(direction * nc_abs, procedure, dof=dof) - target

    lo, hi = 0.0, max(1.0, z_alpha + z_power)
    while gap(hi) < 0.0:
        hi *= 2.0
    return direction * brentq(gap, lo, hi, xtol=1e-12) * se


def validate_relative_lift(relative_lift: float) -> None:
    _require_relative_domain("relative_lift", relative_lift)


def derive_axes_from_baseline(procedure: Any, baseline: Any) -> Any:
    """Derive undeclared analysis axes from the supplied power baseline."""
    if (
        getattr(baseline, "metric_name", None) is not None
        and procedure.metric.metric_type == "mean"
    ):
        procedure = procedure.model_copy(
            update={"metric": procedure.metric.model_copy(update={"metric_type": "quantile"})}
        )
    analysis = procedure.analysis
    analysis_updates: dict[str, str] = {}
    if baseline.compliance != 1.0 and analysis.identification == "randomized":
        analysis_updates["identification"] = "encouragement"
    if baseline.trigger_rate != 1.0 and analysis.population == "assigned":
        analysis_updates["population"] = "triggered"
    if baseline.icc != 0.0 and analysis.variance_adjustment == "none":
        analysis_updates["variance_adjustment"] = "factor_absorption"
    if analysis_updates:
        procedure = procedure.model_copy(
            update={"analysis": analysis.model_copy(update=analysis_updates)}
        )
    already_cuped = any(
        method.variance_reduction != "none"
        for method in (procedure.decision_method, *procedure.sensitivity_methods)
    )
    if baseline.cuped_rho != 0.0 and not already_cuped:
        if procedure.decision_method.conversion_inference == "finite_sample":
            from increment._finite_sample_refusals import refuse_finite_sample_cuped

            refuse_finite_sample_cuped(
                procedure.decision_method.name, adjusted_by="baseline_cuped_rho"
            )
        procedure = procedure.model_copy(
            update={"decision_method": MethodSpec(name="cuped", variance_reduction="cuped")}
        )
    return procedure
