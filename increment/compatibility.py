"""Shared typed compatibility primitives for evidence-family contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field

from increment.errors import (
    CapabilityError,
    CodedError,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)


@dataclass(frozen=True, slots=True)
class UnitFloor:
    minimum_per_arm: int
    kind: Literal["units"] = "units"


@dataclass(frozen=True, slots=True)
class ClusterFloor:
    minimum_per_arm: int
    kind: Literal["clusters"] = "clusters"


@dataclass(frozen=True, slots=True)
class IndependentUnitFloor:
    minimum_total_units: int
    kind: Literal["independent_units"] = "independent_units"


@dataclass(frozen=True, slots=True)
class IndependentBlockFloor:
    """A shared-schedule design's floor: block count, never roster size."""

    minimum_total_blocks: int
    kind: Literal["independent_blocks"] = "independent_blocks"


SamplingFloor = UnitFloor | ClusterFloor | IndependentUnitFloor | IndependentBlockFloor


@dataclass(frozen=True, slots=True)
class Supported:
    """A procedure this package will run, and the terms it runs under.

    Every procedure here estimates its own uncertainty; none is handed a known
    variance. What the three fields below record is therefore the terms, not a
    verdict on the interval: ``assumptions`` states what must hold,
    ``reference`` names the construction the critical value came from, and
    ``floor`` is a coarse declared minimum for compatibility and planning --
    samples below it are refused.

    ``floor`` is neither a calibration threshold nor the estimator's own
    feasibility check. Most arm procedures declare
    ``UnitFloor(minimum_per_arm=2)``, roughly the point where a variance can be
    formed at all; clearing it says nothing about whether the reference
    distribution approximates well. Estimators additionally enforce their own
    bounds, which can sit either side of the declared one: the quantile
    order-statistic bracket needs a sample that depends on both the requested
    quantile and alpha, so a median at 95% is feasible well under the declared
    20 per arm while a 99th percentile needs several hundred.

    Calibration is metric-specific and unreported here. A conversion metric's
    Normal reference degrades with the expected count of events and non-events,
    which no field on this contract carries.
    """

    assumptions: tuple[str, ...]
    floor: SamplingFloor
    reference: str

    def __post_init__(self) -> None:
        # Copy caller-owned lists; freezing the dataclass does not freeze nested values.
        if not isinstance(self.assumptions, tuple):
            object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True, slots=True)
class Unsupported:
    refusal_code: str
    context: Mapping[str, object] = field(default_factory=dict)


Support = Supported | Unsupported


SEQUENTIAL_CUPED_REFUSAL = (
    "CUPED with a coefficient fitted from in-experiment outcomes is admitted only under "
    "asymptotic sequential inference (InferenceSpec(kind='asymptotic_mean')), whose "
    "adjusted_mean and adjusted_ratio_mean laws retain the joint (Y, X) moments that "
    "construction reads: the heteroskedasticity-robust asymptotic regression-adjusted "
    "confidence sequence of Lindon, Ham, Tingley and Bojinov (2022, arXiv:2210.08589). "
    "A unit-summary frame with MetricSpec(covariate=...) and a CUPED method registers it "
    "automatically. The exact Bernoulli e-process route (AlwaysValid) does not admit "
    "CUPED, including a pre-period coefficient. A coefficient and covariate centre "
    "fixed from pre-period data are supported by the asymptotic scalar_mean route "
    "(fit_predeclared_adjustment, declared through InferenceSpec.adjustments or "
    "ScalarMeanModel.adjustment) as a fixed per-unit transform. "
    "Fixed-horizon inference fits the coefficient once at its single look."
)


def _render_compatibility_refusal(
    *,
    code: str,
    reason: str | None = None,
    message: str | None = None,
    **context: object,
) -> str:
    if message is not None:
        return message
    if code == "arm.adjustment.sequential_cuped":
        return SEQUENTIAL_CUPED_REFUSAL
    if code == "arm.inference.cluster":
        return (
            f"sequential inference is not supported with a declared cluster "
            f"('{context.get('cluster')}') -- use fixed-horizon inference (valid for one "
            "planned analysis, not repeated looks)."
        )
    if code == "arm.adjustment.cluster_cuped":
        encouragement = bool(context.get("encouragement"))
        return (
            f"CUPED is not supported with a declared cluster "
            f"('{context.get('cluster')}')"
            + (" under an encouragement design" if encouragement else "")
            + " -- clustered moments carry no per-unit covariate evidence. Drop the "
            "covariate to run the clustered analysis without CUPED (same estimand, "
            "no variance reduction)."
        )
    if code == "arm.adjustment.cluster_prior":
        return (
            f"an informative prior is not supported with a declared cluster "
            f"('{context.get('cluster')}') -- clustered inference uses a t reference."
        )
    if code == "arm.metric.quantile_cluster":
        metric_type = context.get("metric_type", "quantile")
        return (
            f"metric {context.get('metric')!r} (type {metric_type!r}) cannot be estimated "
            f"with a declared cluster ({context.get('cluster')!r}) -- it does not decompose "
            "over the cluster-grain moments the clustered wire format carries."
        )
    if code == "arm.adjustment.sequential_prior":
        metrics = context.get("metrics")
        named = f" (metric(s) {list(metrics)!r})" if isinstance(metrics, (list, tuple)) else ""
        return (
            f"prior and sequential inference are mutually exclusive{named}: each "
            "sequential route's error control is frequentist and route-specific -- "
            "exact for the always-valid e-process, asymptotic for the scalar-mean "
            "route -- and a prior-shifted posterior center would void it. Drop "
            "prior= for these metrics, or use fixed-horizon inference (valid for one "
            "planned analysis, not repeated looks) instead of a sequential kind."
        )
    if code == "arm.metric.quantile_cuped":
        return (
            f"CUPED/variance reduction is not supported for quantile metric "
            f"{context.get('metric')!r} -- a quantile has no mean to adjust; use "
            "variance_reduction='none'"
        )
    if code == "arm.metric.quantile_sequential":
        return (
            f"quantile metric {context.get('metric')!r}: no sequential boundary is defined "
            "for a per-arm order statistic. Use fixed-horizon quantile inference (valid "
            "for one planned analysis, not repeated looks), which needs no sample-size "
            "commitment."
        )
    if (
        code
        in (
            "arm.baseline.iid_cluster_knobs",
            "arm.baseline.cluster_size_required",
            "arm.baseline.absorption_undeclared",
            "arm.baseline.compliance_undeclared",
            "arm.baseline.triggering_undeclared",
            "arm.baseline.cuped_undeclared",
        )
        and "knob" in context
        and "requires" in context
    ):
        return f"{context['knob']}: {context['requires']}"
    details = f" ({reason})" if reason else ""
    if context:
        details += f" context={context!r}"
    return f"unsupported evidence compatibility{details}"


# Public refusal codes let callers record failures without depending on an evidence family.
_COMPATIBILITY_CODES = (
    "arm.metric.unsupported",
    "arm.metric.quantile_cluster",
    "arm.adjustment.sequential_cuped",
    "arm.inference.cluster",
    "arm.adjustment.cluster_cuped",
    "arm.adjustment.cluster_prior",
    "arm.metric.quantile_cuped",
    "arm.metric.quantile_sequential",
    "arm.adjustment.sequential_prior",
    "arm.baseline.iid_cluster_knobs",
    "arm.baseline.cluster_size_required",
    "arm.baseline.absorption_undeclared",
    "arm.baseline.compliance_undeclared",
    "arm.baseline.triggering_undeclared",
    "arm.baseline.cuped_undeclared",
    "contrast.inference",
    "contrast.metric",
    "contrast.decision",
)

# Preserve these exception types so callers can catch invalid requests and
# unsupported methods separately from source capability refusals.
_COMPATIBILITY_ERROR_TYPES: Mapping[str, type[CodedError]] = MappingProxyType(
    {
        "arm.adjustment.sequential_prior": InvalidRequestError,
        "arm.metric.quantile_cuped": UnsupportedRequestError,
        "arm.metric.quantile_sequential": UnsupportedRequestError,
    }
)

# Compatibility codes share one renderer whose text depends on the code, so
# they stay prebuilt RefusalSpecs; the module's own codes use the table.
_REFUSALS: dict[str, RefusalSpec] = {
    _code: RefusalSpec(
        _code,
        _COMPATIBILITY_ERROR_TYPES.get(_code, CapabilityError),
        lambda *, reason=None, message=None, _code=_code, **context: _render_compatibility_refusal(
            code=_code,
            reason=reason,
            message=message,
            **context,
        ),
    )
    for _code in _COMPATIBILITY_CODES
} | refusals(
    InvalidRequestError,
    {
        # `multiplier` is omitted from the context when it is 1, so this one
        # keeps the lambda's own default rather than a template.
        "allocation.alpha.underflow": RefusalSpec(
            "allocation.alpha.underflow",
            InvalidRequestError,
            lambda *, numerator, denominator, multiplier=1: (
                "nominal alpha allocation is below the smallest positive "
                "representable float"
                f" (numerator={numerator!r}, multiplier={multiplier!r}, "
                f"denominator={denominator!r})"
            ),
        ),
        "compatibility.support_unsupported": "support must be Unsupported, got {support_type}",
        "compatibility.denominator": "denominator must be >= 1",
        "compatibility.multiplier": "multiplier must be >= 1",
    },
)
_raise = raiser(_REFUSALS)

# Public read-only view limited to the arm/contrast compatibility catalog;
# _REFUSALS also holds this module's own non-compatibility refusals.
ARM_COMPATIBILITY_REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(
    {code: _REFUSALS[code] for code in _COMPATIBILITY_CODES}
)


def refuse_unsupported(support: Unsupported, /, **context: object) -> NoReturn:
    """Raise a coded capability refusal for an unsupported result."""
    if not isinstance(support, Unsupported):
        _raise("compatibility.support_unsupported", support_type=type(support).__name__)
    _raise(support.refusal_code, **support.context, **context)


def _refuse_alpha_underflow(*, numerator: float, multiplier: int, denominator: int) -> NoReturn:
    context: dict[str, object] = {"numerator": numerator, "denominator": denominator}
    if multiplier != 1:
        context["multiplier"] = multiplier
    _raise("allocation.alpha.underflow", **context)


def _conservative_ratio(numerator: float, multiplier: int, denominator: int) -> float:
    """Return ``numerator * multiplier / denominator`` rounded downward."""
    if denominator < 1:
        _raise("compatibility.denominator")
    if multiplier < 1:
        _raise("compatibility.multiplier")
    if not math.isfinite(numerator) or numerator <= 0.0:
        _refuse_alpha_underflow(
            numerator=numerator,
            multiplier=multiplier,
            denominator=denominator,
        )
    n_num, n_den = numerator.as_integer_ratio()
    exact_num = n_num * multiplier
    exact_denominator = n_den * denominator
    quotient = numerator * multiplier / denominator
    if quotient == 0.0:
        quotient = math.nextafter(0.0, 1.0)

    def _compare(candidate: float) -> int:
        candidate_num, candidate_den = candidate.as_integer_ratio()
        left = candidate_num * exact_denominator
        right = exact_num * candidate_den
        return (left > right) - (left < right)

    while _compare(quotient) > 0:
        quotient = math.nextafter(quotient, 0.0)
    while True:
        next_quotient = math.nextafter(quotient, math.inf)
        if _compare(next_quotient) > 0:
            break
        quotient = next_quotient
    if quotient == 0.0:
        _refuse_alpha_underflow(
            numerator=numerator,
            multiplier=multiplier,
            denominator=denominator,
        )
    return quotient


def _conservative_divide(numerator: float, denominator: int) -> float:
    """Divide and round down to the greatest representable safe float."""
    return _conservative_ratio(numerator, 1, denominator)


class PowerDesign(CodedModel, BaseModel):
    """Numeric solver controls; policy belongs to ``ArmPlanningProcedure``."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    power: float = Field(default=0.80, gt=0.0, lt=1.0, allow_inf_nan=False)
    allocation: float = Field(default=0.5, gt=0.0, lt=1.0, allow_inf_nan=False)


__all__ = [
    "ARM_COMPATIBILITY_REFUSALS",
    "SEQUENTIAL_CUPED_REFUSAL",
    "ClusterFloor",
    "IndependentBlockFloor",
    "IndependentUnitFloor",
    "PowerDesign",
    "SamplingFloor",
    "Support",
    "Supported",
    "UnitFloor",
    "Unsupported",
    "_conservative_divide",
    "refuse_unsupported",
]
