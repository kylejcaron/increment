"""How a persisted row's relative-lift interval was built, for its FCR reissue.

A BH family selects after seeing the data, so every selected interval is reissued at
``fcr_alpha = min(R*q/m, alpha)`` (Benjamini-Yekutieli). ``run_breakout`` reissues it from the
retained moments; a family assembled from finished rows reissues it from the construction the
row persists (``LiftEstimate.reissue``). A construction whose interval cannot be reissued from
persisted state is named and refused, never approximated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate

#: ``unavailable`` rows carry no relative interval (a non-positive arm mean, a degenerate
#: covariance); the additive interval they report, when they have one, is what is reissued.
#: ``conversion`` rows are delta-route conversion rows with their counts, routed again at the
#: corrected level.
Construction = Literal["wald", "conversion", "binomial", "joint", "unavailable"]


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
    if view.delta_evidence is not None:
        return "conversion", "delta-method conversion row with its counts"
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
        case "wald" | "conversion":
            assert view.lift is not None and view.lift.alpha is not None
            alpha_eff = view.lift.alpha
            return alpha_eff if view.alternative == "two-sided" else alpha_eff / 2.0
        case "unavailable":
            assert view.abs_alpha is not None, "classified: the additive interval carries its alpha"
            return view.abs_alpha if view.alternative == "two-sided" else view.abs_alpha / 2.0
        case _:
            assert_never(construction)
