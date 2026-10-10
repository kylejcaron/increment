"""Row-level FCR re-estimation: reissue a selected relative-lift interval at its family level.

A BH family selects after seeing the data, so every selected interval is reissued at
``fcr_alpha = min(R*q/m, alpha)`` (Benjamini-Yekutieli). ``run_breakout`` reissues it from the
retained moments. Here the row's own persisted construction state plays that role, so a family
assembled from finished rows reaches the interval the moment path reaches.

Each construction is rebuilt as its estimator builds it at the requested alpha, then passed
through ``open_bound_from_two_sided_at_target`` exactly as the moment path does, so exact
binomial, Fieller and one-sided geometry are inherited rather than re-derived. A construction
whose interval cannot be reissued from persisted state is named and refused, never approximated.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, assert_never

from increment.estimation._tails import resolvable_expm1, wald_bounds
from increment.estimation.results import (
    BINOMIAL_NUMERICAL_QUALIFICATION,
    BinomialConfidenceSet,
    _alpha_eff_for,
    _fcr_alpha_for,
    _fixed_fcr_parameters,
    _reinverted_binomial_note,
    open_bound_from_two_sided_at_target,
    relative_confidence_set,
)

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate

#: ``unavailable`` rows carry no relative interval (a non-positive arm mean, a degenerate
#: covariance); the additive interval they report, when they have one, is what is reissued.
Construction = Literal["wald", "binomial", "joint", "unavailable"]


def classify(view: LiftEstimate) -> tuple[Construction | None, str]:
    """Name how *view*'s interval was built, or ``None`` with the reason it cannot be reissued."""
    if view.estimand != "itt":
        return None, f"estimand {view.estimand!r}"
    if view.value_scale != "relative":
        return None, "additive-scale row"
    if view.quantile_p_value is not None:
        return None, "quantile interval (its standard error depends on alpha)"
    if view.independent_mean_reference is not None:
        return None, "independent-mean reference"
    if view.confidence_set is not None:
        return None, "percentile-winsorized confidence set (a single-level construction)"
    if view.relative_unavailable_reason is not None:
        if _has_additive_interval(view) and view.abs_alpha is None:
            return None, "additive interval without its persisted alpha"
        return "unavailable", "relative interval unavailable"
    if view.relative_confidence_set is not None:
        return "joint", "joint relative set"
    if view.reference_kind == "binomial":
        return "binomial", "exact binomial set"
    lift = view.lift
    if view.reference_kind not in ("normal", "t") or view.scale != "log" or lift is None:
        return None, f"{view.reference_kind} reference on the {view.scale} scale"
    if lift.open_side is not None:
        return None, "open one-sided interval (the output of a family correction)"
    if lift.alpha is None:
        return None, "interval without its allocated alpha"
    if lift.log_mean is None or lift.log_se is None or not lift.log_se > 0.0:
        return None, "interval without persisted sufficient statistics"
    return "wald", "Wald interval on the log scale"


def _has_additive_interval(view: LiftEstimate) -> bool:
    return view.abs_lb is not None and view.abs_ub is not None


def _additive_reference_df(view: LiftEstimate) -> float | None:
    return view.abs_reference_df if view.abs_reference_kind == "t" else None


def nominal_alpha(view: LiftEstimate, construction: Construction) -> float:
    """The call-level alpha *view* was built at, the cap on its FCR level.

    A directional row carries the doubled display alpha; the cap is the single-tail budget. A row
    with no relative interval reads it from the persisted alpha of its additive interval.
    """
    match construction:
        case "joint":
            assert view.relative_confidence_set is not None
            return view.relative_confidence_set.alpha
        case "binomial":
            assert view.binomial_set is not None
            return view.binomial_set.decision_alpha
        case "wald":
            assert view.lift is not None and view.lift.alpha is not None
            alpha_eff = view.lift.alpha
            return alpha_eff if view.alternative == "two-sided" else alpha_eff / 2.0
        case "unavailable":
            assert view.abs_alpha is not None, "classified: the additive interval carries its alpha"
            return view.abs_alpha if view.alternative == "two-sided" else view.abs_alpha / 2.0
        case _:
            assert_never(construction)


def _wald_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """``infer_lift``'s interval from the persisted constructor moments at *alpha*."""
    from increment.estimation.inference import _resolve_fixed_horizon

    assert view.lift is not None
    parameters = _fixed_fcr_parameters(view)
    assert parameters is not None, "classified: raw statistics are persisted"
    mu, sigma = parameters
    reference = _resolve_fixed_horizon(
        alpha,
        view.alternative,
        dof=view.reference_df if view.reference_kind == "t" else None,
        arm_ns=None,
        prior=None,
    )
    assert reference.crit is not None
    lower, upper = wald_bounds(mu, reference.crit * sigma, 1.0, what="FCR relative interval")
    updates: dict[str, object] = {
        "lift": view.lift.model_copy(
            update={
                "lb": resolvable_expm1(lower, what="FCR relative lower bound"),
                "ub": resolvable_expm1(upper, what="FCR relative upper bound"),
                "level": math.fsum((1.0, -reference.alpha_eff)),
                "alpha": reference.alpha_eff,
            }
        )
    }
    if view.abs_lb is not None or view.abs_ub is not None:
        assert view.abs_diff is not None and view.abs_se is not None
        abs_reference = _resolve_fixed_horizon(
            alpha,
            view.alternative,
            dof=view.abs_reference_df if view.abs_reference_kind == "t" else None,
            arm_ns=None,
            prior=None,
        )
        assert abs_reference.crit is not None
        updates["abs_lb"], updates["abs_ub"] = wald_bounds(
            view.abs_diff, abs_reference.crit, view.abs_se, what="FCR additive interval"
        )
        updates["abs_alpha"] = abs_reference.alpha_eff
    return view.model_copy(update=updates)


def _binomial_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The exact binomial row ``estimate_lift`` builds at *alpha* from the persisted counts."""
    from increment.estimation import binomial_rr
    from increment.estimation.engine import _binomial_abs_bounds

    bset = view.binomial_set
    assert bset is not None
    alpha_eff = alpha if view.alternative == "two-sided" else 2.0 * alpha
    interval = binomial_rr.confidence_interval(
        bset.x_c,
        bset.n_c,
        bset.x_t,
        bset.n_t,
        alpha=alpha,
        alternative=binomial_rr.validate_alternative(view.alternative),
        null_r=1.0 + view.null_lift,
    )
    lower, upper = binomial_rr.to_lift_bounds(interval)
    level = math.fsum((1.0, -alpha_eff))
    updates: dict[str, object] = {
        "binomial_set": BinomialConfidenceSet(
            lower=lower,
            upper=upper,
            alpha=alpha_eff,
            level=level,
            geometry=interval.geometry,
            method=bset.method,
            numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
            x_c=bset.x_c,
            n_c=bset.n_c,
            x_t=bset.x_t,
            n_t=bset.n_t,
            nuisance_beta=binomial_rr.nuisance_beta(alpha),
            decision_alpha=alpha,
        ),
        "note": _reinverted_binomial_note(view.note, interval),
    }
    if view.lift is not None:
        updates["lift"] = view.lift.model_copy(
            update={
                "lb": lower,
                "ub": upper,
                "open_side": "upper" if upper is None else None,
                "level": level,
                "alpha": alpha_eff,
            }
        )
    if view.abs_se is not None:
        assert view.abs_diff is not None
        updates["abs_lb"], updates["abs_ub"] = _binomial_abs_bounds(
            view.abs_diff, view.abs_se, alpha_eff
        )
        updates["abs_alpha"] = alpha_eff
    return view.model_copy(update=updates)


def _joint_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The Fieller row ``infer_ate`` builds at *alpha* from the persisted joint reference."""
    from increment.estimation.inference import _joint_additive_bounds

    relative = view.relative_confidence_set
    assert relative is not None
    reissued = relative_confidence_set(
        relative.reference, alpha=alpha, alternative=view.alternative
    )
    abs_lb, abs_ub = _joint_additive_bounds(
        view.abs_diff,
        view.abs_se,
        reissued.alpha,
        view.alternative,
        view.abs_reference_df if view.abs_reference_kind == "t" else None,
    )
    return view.model_copy(
        update={
            "relative_confidence_set": reissued,
            "lift": reissued.estimate(),
            "abs_lb": abs_lb,
            "abs_ub": abs_ub,
            "abs_alpha": reissued.alpha_eff if abs_lb is not None else None,
        }
    )


def _unavailable_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The central additive interval an estimator cuts for such a row at *alpha*."""
    from increment.estimation.inference import _joint_additive_bounds

    if not _has_additive_interval(view):
        return view
    assert view.abs_diff is not None and view.abs_se is not None
    abs_lb, abs_ub = _joint_additive_bounds(
        view.abs_diff, view.abs_se, alpha, view.alternative, _additive_reference_df(view)
    )
    return view.model_copy(
        update={
            "abs_lb": abs_lb,
            "abs_ub": abs_ub,
            "abs_alpha": _alpha_eff_for(view.alternative, alpha) if abs_lb is not None else None,
        }
    )


def reinterval_at_fcr(
    view: LiftEstimate, construction: Construction, fcr_alpha: float | Fraction
) -> LiftEstimate:
    """*view* reissued at the total noncoverage budget *fcr_alpha*.

    Two-sided rows get a central interval at *fcr_alpha*; a directional row gets the open bound
    ``open_bound_from_two_sided_at_target`` derives from the parent built at the halved alpha. A
    row without a relative interval has its additive interval reissued centrally.
    """
    alpha = _fcr_alpha_for(view.alternative, fcr_alpha)
    match construction:
        case "wald":
            parent = _wald_parent(view, alpha)
        case "binomial":
            parent = _binomial_parent(view, alpha)
        case "joint":
            parent = _joint_parent(view, alpha)
        case "unavailable":
            parent = _unavailable_parent(view, alpha)
        case _:
            assert_never(construction)
    return open_bound_from_two_sided_at_target(parent)


def reinterval_selected(
    view: LiftEstimate, construction: Construction, realized_threshold: float
) -> LiftEstimate:
    """*view* reissued at ``min(realized_threshold, nominal)``, the level a selected row reports.

    A row with no relative and no additive interval has nothing to reissue.
    """
    if construction == "unavailable" and not _has_additive_interval(view):
        return view
    cap = nominal_alpha(view, construction)
    return reinterval_at_fcr(view, construction, min(realized_threshold, cap))
