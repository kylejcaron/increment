"""Refusals every layer that declares or plans ``conversion_inference="finite_sample"`` shares.

A declaration the finite-sample route cannot serve is one hazard whether a definition, a frame
metric, an estimator method or a power plan carries it, so each hazard has one code and one
typed context here. The shared error class is ``InvalidRequestError``: the declaration is an
invalid request for the route it names, wherever it was written.

* ``FINITE_SAMPLE_METRIC_TYPE``: the metric is not a conversion or retention rate
  (context ``metric_type``, and ``metric`` where a metric is named).
* ``FINITE_SAMPLE_CUPED``: the decision is CUPED-adjusted, so the contrast is no longer a pair
  of raw binomial counts (context ``method`` and ``adjusted_by``, the method's own
  ``variance_reduction`` or the planning baseline's ``cuped_rho``).
"""

from __future__ import annotations

from typing import Literal, NoReturn

from increment.errors import InvalidRequestError, refusals, refuse

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "conversion_inference.finite_sample.metric_type": (
            "conversion_inference='finite_sample' applies only to conversion and retention "
            "metrics, and this metric has type {metric_type!r} -- drop it or use "
            "conversion_inference='auto'"
        ),
        "conversion_inference.finite_sample.cuped": (
            "method {method!r} is CUPED-adjusted (through {adjusted_by}) and requests "
            "conversion_inference='finite_sample': CUPED adjusts the outcome by a fitted "
            "covariate slope, so the contrast is no longer a pair of raw binomial counts and "
            "has no finite-sample test inversion. Use conversion_inference='auto' (CUPED then "
            "adjusts on the asymptotic route) or drop the CUPED adjustment"
        ),
    },
)
FINITE_SAMPLE_METRIC_TYPE = _REFUSALS["conversion_inference.finite_sample.metric_type"]
FINITE_SAMPLE_CUPED = _REFUSALS["conversion_inference.finite_sample.cuped"]


def refuse_finite_sample_metric_type(metric_type: str, *, metric: str | None = None) -> NoReturn:
    """Refuse ``finite_sample`` on a metric of ``metric_type`` that is not a conversion or
    retention rate; ``metric`` names it where a request names its metrics."""
    if metric is None:
        refuse(FINITE_SAMPLE_METRIC_TYPE, metric_type=metric_type)
    refuse(FINITE_SAMPLE_METRIC_TYPE, metric_type=metric_type, metric=metric)


def refuse_finite_sample_cuped(
    method: str,
    *,
    adjusted_by: Literal["variance_reduction", "baseline_cuped_rho"] = "variance_reduction",
) -> NoReturn:
    """Refuse ``finite_sample`` on the CUPED-adjusted ``method``."""
    refuse(FINITE_SAMPLE_CUPED, method=method, adjusted_by=adjusted_by)
