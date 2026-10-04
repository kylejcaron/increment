"""Stable refusal registry for readout requests.

Owned below the readout facade so estimators and the request validator share
one spec object per code.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NoReturn

from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    refusals,
    refuse,
)


def _quantile_subject(metric: str | None) -> str:
    return "a quantile metric" if metric is None else f"quantile metric {metric!r}"


def _with_route(message: str, route: str | None) -> str:
    return message if route is None else f"{message}; {route}"


# The table is deliberately small: it owns stable identifiers and exception
# modality, while axis validators decide when a condition applies.  Renderers
# are functions rather than a global condition matrix.
_REFUSALS: dict[str, RefusalSpec] = refusals(
    InvalidRequestError,
    {
        "readout.design.required": "{view}() requires a source with a declared design -- this source was constructed without one (design=None). Pass design= at construction (or control=/control_group= to derive Randomized).",
        "readout.metric.percentile_winsorization": RefusalSpec(
            "readout.metric.percentile_winsorization",
            CapabilityError,
            lambda *, view, names: (
                f"{view}() does not support inference for percentile-winsorized metrics "
                f"({', '.join(names)}) for this design; the supported fixed-horizon path "
                "requires raw independent units and a supported two-sided winsor inference specification"
            ),
        ),
        "readout.metric.daily_winsorization": RefusalSpec(
            "readout.metric.daily_winsorization",
            CapabilityError,
            lambda *, view, names: (
                f"{view} does not support winsorized metrics "
                f"({', '.join(names)}); use run() for inferential reads"
            ),
        ),
        "readout.assignment.estimands": RefusalSpec(
            "readout.assignment.estimands",
            InvalidRequestError,
            lambda *, estimands, mechanism: (
                f"estimands={list(estimands)} requires an encouragement design; "
                f"this design ({mechanism}) identifies only itt -- declare "
                "design: {mechanism: encouragement, uptake: {fact: ...}} on the "
                "experiment to unlock compliance; LATE additionally requires an "
                "exclusion_restriction declaration (see docs/guides/encouragement.md)"
            ),
        ),
        "readout.estimands.unknown": RefusalSpec(
            "readout.estimands.unknown",
            InvalidRequestError,
            lambda *, unknown, supported: (
                f"unknown estimand(s) {sorted(unknown)}; supported: {supported}. "
                "As-treated / per-protocol comparisons are deliberately not offered."
            ),
        ),
        "readout.randomized.value_scale": RefusalSpec(
            "readout.randomized.value_scale",
            UnsupportedRequestError,
            template="value_scale= is an observational-only reporting selector -- the randomized path already reports the absolute axis on every row; declare an absolute margin instead.",
        ),
        "readout.margin.breakout": RefusalSpec(
            "readout.margin.breakout",
            UnsupportedRequestError,
            template="metric(s) {names} declare a non-inferiority margin, but breakout rows are tested two-sided vs 0 -- a per-segment shifted null is not built; the whole-window guardrail read is run()",
        ),
        "readout.value_scale.unknown_metric": RefusalSpec(
            "readout.value_scale.unknown_metric",
            InvalidRequestError,
            lambda *, unknown, selected: (
                "value_scale= names metrics this source does not report on this call: "
                f"{sorted(unknown)!r} (selected: {sorted(selected)!r}) -- refusing rather "
                "than silently dropping the request"
            ),
        ),
        "readout.metric.quantile_breakout": RefusalSpec(
            "readout.metric.quantile_breakout",
            CapabilityError,
            lambda *, metric, route=None, **_: _with_route(
                f"{_quantile_subject(metric)}: breakout dimensions are not supported -- "
                "quantiles do not decompose over segment moments",
                route,
            ),
        ),
        "readout.metric.quantile_alternative": RefusalSpec(
            "readout.metric.quantile_alternative",
            UnsupportedRequestError,
            lambda *, metric, route=None, **_: _with_route(
                f"{_quantile_subject(metric)}: one-sided alternative is not supported "
                "for quantile metrics",
                route,
            ),
        ),
        "readout.observational.prior": "mixture priors are only supported on the relative (log-RR) lift scale served by infer_lift/estimate_lift",
        "readout.value_scale.invalid": "value_scale[{metric!r}]={value_scale!r} must be 'relative' or 'absolute'",
        "readout.value_scale.null": "value_scale='absolute' cannot combine with a non-zero or absolute null for metric {metric!r}",
        "readout.adjustment.absolute_unadjusted": "Method(name='unadjusted') cannot honor value_scale='absolute': the unadjusted moments path already reports the absolute pair alongside relative lift, and its rows are confounded",
        "readout.encouragement.margin": RefusalSpec(
            "readout.encouragement.margin",
            UnsupportedRequestError,
            template="metric(s) {names} declare a non-inferiority margin (Metric.margin/margin_abs, or a plan-bound ExperimentMetric.margin), which applies to the ITT row under an encouragement design -- this request excludes 'itt', so nothing would apply it; add 'itt' to estimands",
        ),
        "readout.run.segment_unsupported": RefusalSpec(
            "readout.run.segment_unsupported",
            UnsupportedRequestError,
            template="metric {metric!r}: run(by=...) is not supported -- the whole-window lift-cell estimator indexes rows by (metric, arm) only, so multiple segments would silently overwrite one another and the result type carries no segment identity -- use breakout()/Analysis.run_breakout() for a per-segment split instead",
        ),
        "readout.arms.no_treatment": RefusalSpec(
            "readout.arms.no_treatment",
            InvalidRequestError,
            lambda *, control_group, observed_arms: (
                f"no treatment arm to compare against control {control_group!r}: the "
                f"loaded moments carry only {sorted(observed_arms)!r}. A lift readout "
                "needs at least one non-control arm; use readouts.srm() to monitor a "
                "treatment-free prefix."
            ),
        ),
        "readout.asof.segment_unsupported": RefusalSpec(
            "readout.asof.segment_unsupported",
            UnsupportedRequestError,
            lambda *, by: (
                f"asof_lift(by={list(by)}) is not supported -- the as-of series result "
                "type (LiftEstimate) carries no segment identity, so every segment's rows "
                "for a date collapse into one bucket and are estimated against the wrong "
                "segment's control -- use Analysis.run_asof_lift(dimension=...) for a "
                "per-segment as-of split instead"
            ),
        ),
        "readout.asof.cluster": RefusalSpec(
            "readout.asof.cluster",
            CapabilityError,
            template="asof_lift() is not supported with a declared cluster ('{cluster}'): the day-axis views have no cluster-robust variance and would silently report unit-grain uncertainty; cluster-robust inference is total-grain only (run()/srm())",
        ),
        "readout.request.readoutrequest_requires_one_resolved": "ReadoutRequest requires one resolved metric configuration per selected metric",
    },
)

READOUT_REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(_REFUSALS)

# One hazard, one code across ingress paths: frame sources, artifacts, definitions and
# the observational readout seam all refuse a quantile metric under this spec.
FRAME_QUANTILE_NO_MOMENTS = RefusalSpec(
    "source.frame.quantile_no_moments",
    CapabilityError,
    template="quantile metric {metric!r} has no moment representation; it is served through unit_frame. {route}",
)


def refuse_quantile_moments(metric: object, design: object) -> NoReturn:
    """Refuse reading a quantile metric as moments, naming the route for *design*.

    An observational design has no quantile estimator: its adjusted-mean
    machinery would otherwise report a mean effect under the quantile metric's name.
    """
    refuse(
        FRAME_QUANTILE_NO_MOMENTS,
        metric=getattr(metric, "name", None),
        route=(
            "an observational design has no quantile estimator; quantile metrics "
            "run under a randomized design"
            if getattr(design, "mechanism", None) == "observational"
            else "use readouts.run, which routes quantiles automatically"
        ),
    )
