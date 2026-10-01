"""Data-generating process for synthetic A/B test experiments.

Every metric shares a per-unit latent effect eta_u ~ N(0, unit_heterogeneity**2).
Conversion and revenue calibrate their treatment shift by root-finding
(Gauss-Hermite quadrature over eta) so the realized lift is exact on the
metric's marginal mean; count's mean shift is a direct closed form.

Conversion (Bernoulli/logit)::

    logit(p_u) = logit(p0) + eta_u + delta_u          p0 = 0.15

Calibrated so ``E[p_treatment] = (1 + true_lift) * E[p_control]``.

Count (Negative Binomial)::

    log(lambda_u) = log(lambda0) + 0.5*eta_u + log(1 + true_lift)   [treatment only]

``lambda0 = 3.0``, dispersion ``phi = 2.0``; lift is multiplicative on the mean.

Revenue (zero-inflated LogNormal), two-part model:

1. Purchase: ``logit(pi_u) = logit(pi0) + 0.3*eta_u + delta_u/2``, ``pi0 = 0.10``.
2. Amount: ``log(Y_u) ~ N(mu + 0.1*eta_u + delta_u/2, sigma**2)``, ``mu = 2.0``,
   ``sigma = 1.0``.

Expected revenue per unit is ``pi * exp(mu + sigma**2/2)``; ``delta`` splits
equally between margin and depth, calibrated so
``E[R_treatment] = (1 + true_lift) * E[R_control]``.

``unit_heterogeneity = 0`` disables eta (identical baseline rates, no
pre/post correlation). No pre-period events are generated here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Literal, NoReturn

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.optimize import brentq

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)


class Scenario(CodedModel, BaseModel):
    """Controls for a synthetic A/B test data generation run.

    ``n_days`` sets only the timestamp spread of generated events, never
    the data volume: conversion and revenue draw at most one event per
    unit for the whole window, and the count metric's mean is fixed
    regardless of window length. Compute per-unit totals, not per-day
    rates, from the output.
    """

    model_config = ConfigDict(frozen=True)

    n_units: int = Field(ge=1)
    n_days: int = Field(ge=1)
    true_lift: dict[str, float]  # metric name -> relative lift

    @field_validator("true_lift")
    @classmethod
    def _supported_true_lifts(cls, values: dict[str, float]) -> dict[str, float]:
        for metric, value in values.items():
            if not math.isfinite(value) or value < -0.99:
                _raise("simulate.dgp.scenario.true_lift_finite", metric=metric, value=value)
        return values

    # Planned fraction assigned to treatment; used as the expected SRM share.
    assignment_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    # Optional realized DGP fraction assigned to treatment. When omitted, the
    # DGP follows the planned ratio while preserving existing scenarios.
    realized_assignment_ratio: float | None = Field(default=None, gt=0.0, lt=1.0)
    # Std dev of the shared per-unit latent effect eta; must be >= 0, with
    # [0, 1] the intended range (0 disables heterogeneity).
    unit_heterogeneity: float = Field(default=0.5, ge=0.0)
    seed: int = 0

    # When uptake_compliance_t and tau_complier are both set, also generates
    # uptake + compliance-driven outcome events (see _build_uptake_events).
    uptake_compliance_t: float | None = Field(default=None, ge=0.0, le=1.0)
    uptake_compliance_c: float = Field(default=0.0, ge=0.0, le=1.0)
    tau_complier: float | None = None

    # uptake_confounding and outcome_confounding tilt uptake/outcome by the
    # shared latent eta (see _build_uptake_events); nonzero values bias as-treated, not ITT.
    uptake_confounding: float = 0.0
    outcome_confounding: float = 0.0

    # Number of i.i.d. pre-treatment segments, observable via a "segment"
    # event carrying the segment index (0..n_segments-1); independent of
    # eta/treatment/group. 1 (default) disables segmentation -- no
    # "segment" event is emitted, so existing scenarios are unaffected.
    n_segments: int = Field(default=1, ge=1)

    # Std dev of an observable pre-treatment covariate w ~ N(0, covariate_sd**2),
    # independent of eta and segment, emitted via a "covariate" event.
    # 0 (default) disables it -- no "covariate" event is emitted.
    covariate_sd: float = Field(default=0.0, ge=0.0)

    # Interaction scales each metric's treatment shift by covariate ``w``, so
    # per-unit effects vary and CATE/breakout estimators can recover them.
    # Exact-mean corrections in ``_build_*_events`` keep the average lift at
    # ``true_lift``; only the per-unit spread changes.
    covariate_interaction: float = 0.0


# ---------------------------------------------------------------------------
# Distributional helpers

_INV_LOGIT_MAX = 35.0  # clip to avoid overflow in exp


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _inv_logit_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised inverse logit with clipping."""
    clipped = np.clip(x, -_INV_LOGIT_MAX, _INV_LOGIT_MAX)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit_safe(p: float) -> float:
    """Logit extended to [0, 1]: maps 0 -> -inf and 1 -> +inf, so a
    degenerate probability (exactly 0 or 1) stays exact under any finite
    additive shift.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return math.log(p / (1.0 - p))


# Gauss-Hermite (probabilists') quadrature nodes/weights for expectations
# over eta ~ N(0, s^2): E[f(eta)] ~= weighted average of f(s * node).
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(64)


def _conversion_logit_shift(
    lift: float, p0: float, het_sd: float, treat_extra_var: float = 0.0
) -> float:
    """Treatment logit shift that realizes ``lift`` on the marginal rate.

    Solves for ``delta`` such that::

        E[invlogit(logit(p0) + delta + eta + w)] = (1 + lift) * E[invlogit(logit(p0) + eta)]

    with ``eta ~ N(0, het_sd**2)`` and, when ``treat_extra_var`` is set, an
    independent treatment-only interaction term ``w`` folded in as
    ``eta + w ~ N(0, het_sd**2 + treat_extra_var)`` -- the sum of two
    independent zero-mean normals is itself normal, so this stays a 1D
    Gauss-Hermite quadrature. A fixed odds shift of ``log(1 + lift)``
    would undershoot the rate lift since the sigmoid is nonlinear;
    solving directly keeps it exact.

    Raises ``ValueError`` if ``(1 + lift) * p_control >= 1`` (the
    requested rate lift is unattainable for a probability metric).
    """
    if lift == 0.0 and treat_extra_var == 0.0:
        return 0.0
    x_control = _logit(p0) + het_sd * _GH_NODES
    p_control = float(np.average(_inv_logit_vec(x_control), weights=_GH_WEIGHTS))
    target = (1.0 + lift) * p_control
    if target >= 1.0:
        _raise(
            "simulate.dgp.conversion_lift_unattainable",
            lift=lift,
            target=target,
            p_control=p_control,
        )
    treat_sd = math.sqrt(het_sd**2 + treat_extra_var)
    x_treat = _logit(p0) + treat_sd * _GH_NODES

    def gap(delta: float) -> float:
        return float(np.average(_inv_logit_vec(x_treat + delta), weights=_GH_WEIGHTS)) - target

    return float(brentq(gap, -_INV_LOGIT_MAX, _INV_LOGIT_MAX))


def _revenue_lift_shift(lift: float, pi0: float, het_sd: float) -> float:
    """Total treatment shift that realizes ``lift`` on marginal mean revenue.

    Splits ``delta`` equally between purchase log-odds and amount log-mean,
    then solves::

        E[pi_t(eta) * exp(delta/2 + 0.1*eta)] = (1 + lift) * E[pi_c(eta) * exp(0.1*eta)]

    with ``eta ~ N(0, het_sd**2)`` by Gauss-Hermite quadrature (the
    arm-invariant lognormal factor cancels from both sides). A naive
    half-split of ``log(1 + lift)`` would undershoot the mean lift;
    root-finding keeps it exact.
    """
    if lift == 0.0:
        return 0.0
    pi_arg = _logit(pi0) + 0.3 * het_sd * _GH_NODES
    amount_w = np.exp(0.1 * het_sd * _GH_NODES)
    mean_control = float(np.average(_inv_logit_vec(pi_arg) * amount_w, weights=_GH_WEIGHTS))
    target = (1.0 + lift) * mean_control

    def gap(delta: float) -> float:
        half = delta / 2.0
        mean_treat = math.exp(half) * float(
            np.average(_inv_logit_vec(pi_arg + half) * amount_w, weights=_GH_WEIGHTS)
        )
        return mean_treat - target

    return float(brentq(gap, -80.0, 80.0))


# ---------------------------------------------------------------------------
# Event generation helpers


_EVENT_SCHEMA = pa.schema(
    [
        ("unit_id", pa.string()),
        ("ts", pa.timestamp("us")),
        ("event", pa.string()),
        ("experiment_id", pa.string()),
        ("group_id", pa.string()),
        ("value", pa.float64()),
    ]
)

_START_DATE = datetime(2025, 1, 1)


def _empty_table() -> pa.Table:
    return pa.table(
        {
            k: pa.array([], type=t)
            for k, t in zip(_EVENT_SCHEMA.names, _EVENT_SCHEMA.types, strict=True)
        },
        schema=_EVENT_SCHEMA,
    )


def _mk_ts(day_offset: int, hour: int = 0, minute: int = 0) -> datetime:
    return _START_DATE + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))


def _build_exposure_events(
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    rng: np.random.Generator,
    experiment_id: str = "sim",
) -> tuple[pa.Table, np.ndarray]:
    """Generate one exposure row per unit on the experiment start date.

    Returns the exposure event table plus each unit's own exposure
    timestamp (object array of ``datetime``, same order as *unit_ids*) --
    the anchor every post-exposure event builder stamps its own events
    relative to, so a sampled day-zero event never predates the unit's
    actual exposure.
    """
    n = len(unit_ids)
    hours = rng.integers(0, 6, size=n)
    minutes = rng.integers(0, 60, size=n)
    ts_list = [_mk_ts(0, int(h), int(m)) for h, m in zip(hours, minutes, strict=True)]
    exposure_ts = np.array(ts_list, dtype=object)
    nan_arr = pa.array([float("nan")] * n, type=pa.float64())
    table = pa.table(
        {
            "unit_id": pa.array(unit_ids, type=pa.string()),
            "ts": pa.array(ts_list, type=pa.timestamp("us")),
            "event": pa.array(["exposure"] * n, type=pa.string()),
            "experiment_id": pa.array([experiment_id] * n, type=pa.string()),
            "group_id": pa.array(group_ids, type=pa.string()),
            "value": nan_arr,
        },
        schema=_EVENT_SCHEMA,
    )
    return table, exposure_ts


def _post_exposure_timestamps(
    exposure_ts: np.ndarray, n_days: int, rng: np.random.Generator
) -> list[datetime]:
    """One post-exposure timestamp per element of *exposure_ts*.

    Each event lands at its own unit's actual exposure timestamp plus a
    nonnegative sampled day offset (``[0, n_days)``, preserving the
    existing sampled-day/window contract) and a strictly positive
    within-day increment (``(0, 86400)`` seconds) -- so a day-zero event
    (``day offset == 0``) still lands strictly after exposure instead of
    at the independent midnight timestamp the previous implementation
    used, which could predate a unit's own (randomized) exposure time.
    """
    n = len(exposure_ts)
    if n == 0:
        return []
    day_offset = rng.integers(0, n_days, size=n)
    within_day_seconds = rng.integers(1, 86_400, size=n)  # in (0, 1 day), never zero
    # Accepts object arrays of ``datetime`` and ``datetime64`` arrays alike;
    # ``tolist`` on microsecond resolution yields ``datetime`` objects.
    start = np.asarray(exposure_ts, dtype="datetime64[us]")
    day_offset *= 86_400
    day_offset += within_day_seconds
    return (start + day_offset.view("timedelta64[s]")).tolist()


def _build_segment_events(
    unit_ids: np.ndarray, group_ids: np.ndarray, segment: np.ndarray
) -> pa.Table:
    """One observable "segment" row per unit: the pre-treatment segment
    index (0..n_segments-1), independent of eta and treatment -- the
    breakout dimension the simulation runner routes through
    ``readouts.breakout``.
    """
    n = len(unit_ids)
    return pa.table(
        {
            "unit_id": pa.array(unit_ids, type=pa.string()),
            "ts": pa.array([_mk_ts(-1)] * n, type=pa.timestamp("us")),
            "event": pa.array(["segment"] * n, type=pa.string()),
            "experiment_id": pa.array([""] * n, type=pa.string()),
            "group_id": pa.array(group_ids, type=pa.string()),
            "value": pa.array(segment.astype(float), type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )


def _build_covariate_events(
    unit_ids: np.ndarray, group_ids: np.ndarray, covariate: np.ndarray
) -> pa.Table:
    """One observable "covariate" row per unit: a pre-treatment continuous
    covariate ``w``, independent of eta -- the CATE interaction target the
    simulation runner routes through ``increment.cate.estimate_cate``.
    """
    n = len(unit_ids)
    return pa.table(
        {
            "unit_id": pa.array(unit_ids, type=pa.string()),
            "ts": pa.array([_mk_ts(-1)] * n, type=pa.timestamp("us")),
            "event": pa.array(["covariate"] * n, type=pa.string()),
            "experiment_id": pa.array([""] * n, type=pa.string()),
            "group_id": pa.array(group_ids, type=pa.string()),
            "value": pa.array(covariate, type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )


def _build_conversion_events(
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    eta: np.ndarray,
    lift: float,
    p0: float,
    het_sd: float,
    rng: np.random.Generator,
    n_days: int,
    *,
    exposure_ts: np.ndarray,
    w: np.ndarray | None = None,
    covariate_interaction: float = 0.0,
    covariate_sd: float = 0.0,
) -> pa.Table:
    """Bernoulli conversion events.

    ``lift`` is the relative lift of the marginal conversion rate: the
    treatment arm gets a logit shift (``_conversion_logit_shift``) so
    ``E[p_treatment] = (1 + lift) * E[p_control]``, expectation over the
    unit random effect ``eta`` (scale ``het_sd``). A nonzero
    ``covariate_interaction`` additionally shifts each treated unit's
    logit by ``covariate_interaction * w_u``; the calibration folds that
    extra variance in (see ``_conversion_logit_shift``), so the marginal
    rate lift stays exact.

    Each event is stamped from its own unit's *exposure_ts* plus a
    sampled nonnegative day offset and a positive within-day increment
    (see ``_post_exposure_timestamps``), so it always lands strictly
    after that unit's actual exposure, including a day-zero offset.
    """
    treat_extra_var = (covariate_interaction * covariate_sd) ** 2
    delta = _conversion_logit_shift(lift, p0, het_sd, treat_extra_var)

    logit_control = _logit(p0) + eta
    logit_treat = logit_control + delta
    if covariate_interaction != 0.0 and w is not None:
        logit_treat = logit_treat + covariate_interaction * w
    p = np.where(
        group_ids == "treatment", _inv_logit_vec(logit_treat), _inv_logit_vec(logit_control)
    )

    converts = rng.binomial(1, p).astype(bool)
    n_conv = int(converts.sum())
    if n_conv == 0:
        return _empty_table()

    conv_units = unit_ids[converts]
    ts_list = _post_exposure_timestamps(exposure_ts[converts], n_days, rng)

    return pa.table(
        {
            "unit_id": pa.array(conv_units, type=pa.string()),
            "ts": pa.array(ts_list, type=pa.timestamp("us")),
            "event": pa.array(["conversion"] * n_conv, type=pa.string()),
            "experiment_id": pa.array([""] * n_conv, type=pa.string()),
            "group_id": pa.array(group_ids[converts].tolist(), type=pa.string()),
            "value": pa.array([1.0] * n_conv, type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )


def _raise_count_rate(*, reason: str, **context: object) -> NoReturn:
    controls = ("true_lift.count", "unit_heterogeneity", "covariate_interaction")
    route = (
        "Reduce the magnitude of count lift, unit heterogeneity or covariate interaction "
        "to keep per-unit count rates representable."
    )
    if reason == "negative_binomial_rate_underflow":
        route = (
            "Reduce negative log-rate shifts from count lift, unit heterogeneity or "
            "covariate interaction so sampled count rates remain representable."
        )
    if reason.startswith("event_"):
        controls = ("n_units", *controls)
        route = (
            "Reduce n_units or positive count lift; reduce unit heterogeneity or "
            "covariate interaction if they create very large per-unit counts."
        )
    _raise(
        "simulate.dgp.count_rate_unrepresentable",
        reason=reason,
        controls=controls,
        route=route,
        **context,
    )


def _count_means(
    group_ids: np.ndarray,
    eta: np.ndarray,
    lift: float,
    base_rate: float,
    *,
    w: np.ndarray | None = None,
    covariate_interaction: float = 0.0,
    covariate_sd: float = 0.0,
) -> np.ndarray:
    """Return the declared per-unit count means without clipping."""
    lift_log = math.log1p(lift)
    if covariate_interaction != 0.0:
        lift_log -= 0.5 * (covariate_interaction * covariate_sd) ** 2

    log_lambda = math.log(base_rate) + 0.5 * np.asarray(eta, dtype=float)
    treat_mask = np.asarray(group_ids) == "treatment"
    log_lambda[treat_mask] += lift_log
    if covariate_interaction != 0.0 and w is not None:
        log_lambda[treat_mask] += covariate_interaction * np.asarray(w)[treat_mask]

    if not np.all(np.isfinite(log_lambda)):
        _raise_count_rate(
            reason="nonfinite_log_mean",
            min_log_mean=float(np.nanmin(log_lambda)),
            max_log_mean=float(np.nanmax(log_lambda)),
        )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        lam = np.exp(log_lambda)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0.0):
        _raise_count_rate(
            reason="float_parameter_range",
            min_log_mean=float(np.min(log_lambda)),
            max_log_mean=float(np.max(log_lambda)),
        )
    return lam


_ARROW_STRING_MAX_BYTES = (1 << 31) - 1


def _check_count_string_capacity(
    unit_ids: np.ndarray, group_ids: np.ndarray, counts: np.ndarray, ne: int
) -> None:
    """Check repeated UTF-8 data before expanding Arrow string columns."""
    if ne > _ARROW_STRING_MAX_BYTES // 5:
        _raise_count_rate(
            reason="event_string_data_capacity",
            column="event",
            total_string_bytes=ne * 5,
            max_string_bytes=_ARROW_STRING_MAX_BYTES,
        )
    for column, values in (("unit_id", unit_ids), ("group_id", group_ids)):
        if values.dtype.kind in "US" and ne * values.dtype.itemsize <= _ARROW_STRING_MAX_BYTES:
            continue
        lengths = pc.call_function("binary_length", [pa.array(values, type=pa.string())]).to_numpy()
        # ne <= int32_max / 5 and every length <= int32_max: the dot fits int64.
        total_bytes = int(np.dot(counts.astype(np.int64, copy=False), lengths))
        if total_bytes > _ARROW_STRING_MAX_BYTES:
            _raise_count_rate(
                reason="event_string_data_capacity",
                column=column,
                total_string_bytes=total_bytes,
                max_string_bytes=_ARROW_STRING_MAX_BYTES,
            )


def _build_count_events(
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    eta: np.ndarray,
    lift: float,
    base_rate: float,
    dispersion: float,
    rng: np.random.Generator,
    n_days: int,
    *,
    exposure_ts: np.ndarray,
    w: np.ndarray | None = None,
    covariate_interaction: float = 0.0,
    covariate_sd: float = 0.0,
) -> pa.Table:
    """Negative-Binomial count events ("visits"). Each unit draws
    ``Y_u ~ NegBinom(mean=lambda_u, dispersion)`` scheduled on random days.
    Lift is multiplicative on the mean.
    """
    n = len(unit_ids)
    lam = _count_means(
        group_ids,
        eta,
        lift,
        base_rate,
        w=w,
        covariate_interaction=covariate_interaction,
        covariate_sd=covariate_sd,
    )

    # A unit-mean Gamma multiplier gives Var(Y) = lambda + lambda**2/r
    # without quantizing tiny means through a probability rounded near one.
    try:
        gamma_rates = np.asarray(rng.gamma(dispersion, 1.0 / dispersion, size=n))
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            rates = gamma_rates * lam
    except (FloatingPointError, OverflowError, ValueError) as exc:
        _raise_count_rate(
            reason="negative_binomial_sampler",
            min_mean=float(np.min(lam)),
            max_mean=float(np.max(lam)),
            sampler_error=type(exc).__name__,
        )
    collapsed = (gamma_rates > 0.0) & (lam > 0.0) & (rates == 0.0)
    if np.any(collapsed):
        _raise_count_rate(
            reason="negative_binomial_rate_underflow",
            min_mean=float(np.min(lam)),
            max_mean=float(np.max(lam)),
            collapsed_rates=int(np.count_nonzero(collapsed)),
            min_gamma=float(np.min(gamma_rates[collapsed])),
        )
    try:
        counts = np.asarray(rng.poisson(rates), dtype=np.int64)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        _raise_count_rate(
            reason="negative_binomial_sampler",
            min_mean=float(np.min(lam)),
            max_mean=float(np.max(lam)),
            sampler_error=type(exc).__name__,
        )
    if np.any(counts < 0):
        _raise_count_rate(
            reason="negative_count",
            min_mean=float(np.min(lam)),
            max_mean=float(np.max(lam)),
        )

    # Use a fixed-width sum only when its bound is provably safe; otherwise
    # fall back to Python integers so totals cannot wrap before ``repeat``.
    uint_max = int(np.iinfo(np.uint64).max)
    max_count = int(counts.max(initial=0))
    if max_count == 0 or n <= uint_max // max_count:
        ne = int(counts.sum(dtype=np.uint64))
    else:
        ne = sum(int(count) for count in counts)
    if ne > np.iinfo(np.intp).max:
        _raise_count_rate(
            reason="event_index_capacity",
            total_events=ne,
            max_indexed_events=int(np.iinfo(np.intp).max),
        )
    if ne == 0:
        return _empty_table()
    itemsize = max(unit_ids.dtype.itemsize, group_ids.dtype.itemsize, exposure_ts.dtype.itemsize)
    if itemsize and ne > np.iinfo(np.intp).max // itemsize:
        _raise_count_rate(
            reason="event_array_byte_capacity",
            total_events=ne,
            bytes_per_event=itemsize,
            max_array_bytes=int(np.iinfo(np.intp).max),
        )
    _check_count_string_capacity(unit_ids, group_ids, counts, ne)

    # One row per event: repeat each unit id by its count, then draw all
    # event days in one call (iid Uniform over the window).  Timestamp and
    # Arrow materialization are part of the same representability boundary.
    try:
        rep_units = np.repeat(unit_ids, counts)
        rep_groups = np.repeat(group_ids, counts)
        rep_exposure_ts = np.repeat(exposure_ts, counts)
        ts_list = _post_exposure_timestamps(rep_exposure_ts, n_days, rng)
        return pa.table(
            {
                "unit_id": pa.array(rep_units, type=pa.string()),
                "ts": pa.array(ts_list, type=pa.timestamp("us")),
                "event": pa.array(["visit"] * ne, type=pa.string()),
                "experiment_id": pa.array([""] * ne, type=pa.string()),
                "group_id": pa.array(rep_groups, type=pa.string()),
                "value": pa.array([1.0] * ne, type=pa.float64()),
            },
            schema=_EVENT_SCHEMA,
        )
    except (MemoryError, OverflowError, pa.ArrowCapacityError) as exc:
        _raise_count_rate(
            reason="event_materialization",
            total_events=ne,
            allocation_error=type(exc).__name__,
        )


# Cohesive synthetic revenue generation; splitting this path is deferred.
def _build_revenue_events(  # noqa: PLR0913
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    eta: np.ndarray,
    lift: float,
    pi0: float,
    log_mu: float,
    log_sigma: float,
    het_sd: float,
    rng: np.random.Generator,
    n_days: int,
    *,
    exposure_ts: np.ndarray,
    w: np.ndarray | None = None,
    covariate_interaction: float = 0.0,
    covariate_sd: float = 0.0,
) -> pa.Table:
    """Zero-inflated LogNormal revenue events: purchase probability pi_u
    (logit) times positive amount Y_u ~ LogNormal(mu_u, sigma**2); expected
    revenue per unit is ``pi * exp(mu + sigma**2/2)``. The treatment shift
    splits equally between margin and depth, calibrated
    (``_revenue_lift_shift``) so
    ``E[R_treatment] = (1 + lift) * E[R_control]``.

    A nonzero ``covariate_interaction`` additionally shifts each treated
    unit's log-amount (spend depth, not purchase odds) by
    ``covariate_interaction * w_u``, minus its lognormal-mean correction
    -- ``w`` is independent of ``eta`` and mean zero, so the correction
    keeps the realized average revenue lift exact.
    """
    lift_half = _revenue_lift_shift(lift, pi0, het_sd) / 2.0

    # Purchase probability with lift on odds
    logit_pi = _logit(pi0) + 0.3 * eta
    treat_mask = group_ids == "treatment"
    logit_pi[treat_mask] += lift_half
    pi = _inv_logit_vec(np.clip(logit_pi, -_INV_LOGIT_MAX, _INV_LOGIT_MAX))

    # Amount: shift log-mean
    log_mu_adj = log_mu + 0.1 * eta
    log_mu_adj[treat_mask] += lift_half
    if covariate_interaction != 0.0 and w is not None:
        correction = 0.5 * (covariate_interaction * covariate_sd) ** 2
        log_mu_adj[treat_mask] += covariate_interaction * w[treat_mask] - correction

    purchases = rng.binomial(1, pi).astype(bool)
    n_buy = int(purchases.sum())
    if n_buy == 0:
        return _empty_table()

    buy_units = unit_ids[purchases]
    ts_list = _post_exposure_timestamps(exposure_ts[purchases], n_days, rng)
    amounts = np.exp(rng.normal(log_mu_adj[purchases], log_sigma))

    return pa.table(
        {
            "unit_id": pa.array(buy_units, type=pa.string()),
            "ts": pa.array(ts_list, type=pa.timestamp("us")),
            "event": pa.array(["revenue"] * n_buy, type=pa.string()),
            "experiment_id": pa.array([""] * n_buy, type=pa.string()),
            "group_id": pa.array(group_ids[purchases].tolist(), type=pa.string()),
            "value": pa.array(amounts.tolist(), type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )


def _build_uptake_events(
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    eta: np.ndarray,
    compliance_t: float,
    compliance_c: float,
    uptake_confounding: float,
    rng: np.random.Generator,
    n_days: int,
    *,
    exposure_ts: np.ndarray,
) -> tuple[pa.Table, np.ndarray]:
    """Bernoulli uptake fact for an encouragement design.

    Treatment units take up with probability ``compliance_t``, control
    with ``compliance_c`` (0 = one-sided, no-defiers). ``uptake_confounding``
    tilts each unit's propensity by its latent effect ``eta``::

        p_u = invlogit(logit(compliance_arm) + uptake_confounding * eta)

    so uptake selects on the same latent driving ``outcome_confounding``.
    Assignment stays randomized independent of ``eta``, so intent-to-treat
    is unbiased even though as-treated becomes biased. A nonzero tilt also
    shifts the marginal uptake rate off the compliance parameter; 0/1
    compliance stays deterministic regardless of ``eta``.

    Returns the uptake table plus the realized 0/1 indicator per unit.
    """
    if uptake_confounding == 0.0:
        # No tilt: use compliance directly (routing through logit/invlogit
        # would perturb it by float roundoff).
        p = np.where(group_ids == "treatment", compliance_t, compliance_c)
    else:
        base_logit = np.where(
            group_ids == "treatment",
            _logit_safe(compliance_t),
            _logit_safe(compliance_c),
        )
        shifted = base_logit + uptake_confounding * eta
        p = _inv_logit_vec(shifted)
        # Degenerate compliance (exactly 0 or 1) must stay exact - clipping
        # inside _inv_logit_vec would otherwise leave a ~1e-15 residual probability.
        p = np.where(np.isneginf(shifted), 0.0, p)
        p = np.where(np.isposinf(shifted), 1.0, p)
    d = rng.binomial(1, p).astype(bool)
    n_up = int(d.sum())
    if n_up == 0:
        return _empty_table(), d.astype(float)

    up_units = unit_ids[d]
    ts_list = _post_exposure_timestamps(exposure_ts[d], n_days, rng)
    return (
        pa.table(
            {
                "unit_id": pa.array(up_units, type=pa.string()),
                "ts": pa.array(ts_list, type=pa.timestamp("us")),
                "event": pa.array(["uptake"] * n_up, type=pa.string()),
                "experiment_id": pa.array([""] * n_up, type=pa.string()),
                "group_id": pa.array(group_ids[d].tolist(), type=pa.string()),
                "value": pa.array([1.0] * n_up, type=pa.float64()),
            },
            schema=_EVENT_SCHEMA,
        ),
        d.astype(float),
    )


def _build_encouragement_outcome_events(
    unit_ids: np.ndarray,
    group_ids: np.ndarray,
    d: np.ndarray,
    eta: np.ndarray,
    tau_complier: float,
    base: float,
    noise_sd: float,
    outcome_confounding: float,
    rng: np.random.Generator,
    n_days: int,
    *,
    exposure_ts: np.ndarray,
) -> pa.Table:
    """Continuous outcome ``y = base + outcome_confounding * eta +
    tau_complier * d + noise``, one row per unit.

    ``d`` is the realized uptake indicator (not assignment), so the effect
    is driven by compliance. ``outcome_confounding`` routes the shared
    latent ``eta`` into the outcome; combined with a nonzero uptake tilt on
    the same latent, uptake becomes informative about the outcome baseline
    while randomized assignment keeps intent-to-treat unbiased.
    """
    n = len(unit_ids)
    noise = rng.normal(0.0, noise_sd, size=n)
    y = base + outcome_confounding * eta + tau_complier * d + noise
    ts_list = _post_exposure_timestamps(exposure_ts, n_days, rng)
    return pa.table(
        {
            "unit_id": pa.array(unit_ids, type=pa.string()),
            "ts": pa.array(ts_list, type=pa.timestamp("us")),
            "event": pa.array(["outcome"] * n, type=pa.string()),
            "experiment_id": pa.array([""] * n, type=pa.string()),
            "group_id": pa.array(group_ids.tolist(), type=pa.string()),
            "value": pa.array(y.tolist(), type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Public API

# Default distributional parameters
_CONVERSION_P0 = 0.15
_COUNT_BASE_RATE = 3.0
_COUNT_DISPERSION = 2.0
_REVENUE_PI0 = 0.10
_REVENUE_LOG_MU = 2.0
_REVENUE_LOG_SIGMA = 1.0
_ENCOURAGEMENT_BASE = 10.0  # outcome baseline (control, non-complier level)
_ENCOURAGEMENT_NOISE_SD = 2.0

# Known metric names and their event types
_SUPPORTED_METRICS = frozenset({"conversion", "count", "revenue"})


def simulate_raw_logs(scenario: Scenario) -> pa.Table:
    """Generate a synthetic raw event log for a full A/B experiment.

    Returns a single ``pa.Table`` (one row per event) with columns:

    * ``unit_id`` - experimental unit id.
    * ``ts`` - event timestamp.
    * ``event`` - ``"exposure"``, ``"conversion"``, ``"visit"``, ``"revenue"``,
      (encouragement designs only) ``"uptake"`` / ``"outcome"``, or
      (``n_segments > 1`` / ``covariate_sd > 0``) ``"segment"`` / ``"covariate"``.
    * ``experiment_id`` - set on exposure rows, empty elsewhere.
    * ``group_id`` - ``"control"`` or ``"treatment"``.
    * ``value`` - numeric metric value (NaN for exposures; the segment
      index or covariate draw for those two observable rows).
    """
    unknown = set(scenario.true_lift) - _SUPPORTED_METRICS
    if unknown:
        _raise(
            "simulate.dgp.unsupported_metric_supported",
            unknown=sorted(unknown),
            supported=sorted(_SUPPORTED_METRICS),
        )

    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_units

    # Unit IDs
    unit_ids = np.array([f"u{i:06d}" for i in range(n)])

    # Per-unit random effects (shared latent factor across all metrics)
    eta = rng.normal(0, scenario.unit_heterogeneity, size=n)
    realized_ratio = (
        scenario.assignment_ratio
        if scenario.realized_assignment_ratio is None
        else scenario.realized_assignment_ratio
    )
    # Group assignment uses realized ratio; SRM expectations use planned ratio.
    group = rng.binomial(1, realized_ratio, size=n)
    group_ids = np.where(group == 0, "control", "treatment")

    # --- Exposure events ---
    exposures, exposure_ts = _build_exposure_events(unit_ids, group_ids, rng)

    # --- Metric events ---
    tables = [exposures]

    # --- Observable pre-treatment segment/covariate ---
    # Only drawn (and only advances rng) when the scenario actually uses
    # them, so a default scenario's other draws are unaffected.
    segment_idx = rng.integers(0, scenario.n_segments, size=n) if scenario.n_segments > 1 else None
    w = rng.normal(0, scenario.covariate_sd, size=n) if scenario.covariate_sd > 0 else None

    for metric_name, rel_lift in scenario.true_lift.items():
        if metric_name == "conversion":
            tbl = _build_conversion_events(
                unit_ids,
                group_ids,
                eta,
                rel_lift,
                _CONVERSION_P0,
                scenario.unit_heterogeneity,
                rng,
                scenario.n_days,
                exposure_ts=exposure_ts,
                w=w,
                covariate_interaction=scenario.covariate_interaction,
                covariate_sd=scenario.covariate_sd,
            )
        elif metric_name == "count":
            tbl = _build_count_events(
                unit_ids,
                group_ids,
                eta,
                rel_lift,
                _COUNT_BASE_RATE,
                _COUNT_DISPERSION,
                rng,
                scenario.n_days,
                exposure_ts=exposure_ts,
                w=w,
                covariate_interaction=scenario.covariate_interaction,
                covariate_sd=scenario.covariate_sd,
            )
        elif metric_name == "revenue":
            tbl = _build_revenue_events(
                unit_ids,
                group_ids,
                eta,
                rel_lift,
                _REVENUE_PI0,
                _REVENUE_LOG_MU,
                _REVENUE_LOG_SIGMA,
                scenario.unit_heterogeneity,
                rng,
                scenario.n_days,
                exposure_ts=exposure_ts,
                w=w,
                covariate_interaction=scenario.covariate_interaction,
                covariate_sd=scenario.covariate_sd,
            )
        else:
            _raise("simulate.dgp.unknown_metric_true", metric_name=metric_name)

        tables.append(tbl)

    if segment_idx is not None:
        tables.append(_build_segment_events(unit_ids, group_ids, segment_idx))
    if w is not None:
        tables.append(_build_covariate_events(unit_ids, group_ids, w))

    # --- Encouragement design: uptake + compliance-driven outcome ---
    if scenario.uptake_compliance_t is not None or scenario.tau_complier is not None:
        if scenario.uptake_compliance_t is None or scenario.tau_complier is None:
            _raise("simulate.dgp.uptake_compliance_t")
        uptake_tbl, d = _build_uptake_events(
            unit_ids,
            group_ids,
            eta,
            scenario.uptake_compliance_t,
            scenario.uptake_compliance_c,
            scenario.uptake_confounding,
            rng,
            scenario.n_days,
            exposure_ts=exposure_ts,
        )
        outcome_tbl = _build_encouragement_outcome_events(
            unit_ids,
            group_ids,
            d,
            eta,
            scenario.tau_complier,
            _ENCOURAGEMENT_BASE,
            _ENCOURAGEMENT_NOISE_SD,
            scenario.outcome_confounding,
            rng,
            scenario.n_days,
            exposure_ts=exposure_ts,
        )
        tables.append(uptake_tbl)
        tables.append(outcome_tbl)

    combined = pa.concat_tables(tables)
    # Sort by (unit_id, ts) for reproducibility
    unit_col = combined.column("unit_id").to_numpy()
    ts_col = combined.column("ts").to_numpy()
    idx = np.lexsort((ts_col, unit_col))
    combined = combined.take(idx)
    return combined


def _validate_binomial_fixture_counts(x_c: int, n_c: int, x_t: int, n_t: int) -> None:
    if (
        not all(isinstance(v, int) for v in (x_c, n_c, x_t, n_t))
        or n_c < 1
        or n_t < 1
        or not (0 <= x_c <= n_c)
        or not (0 <= x_t <= n_t)
    ):
        _raise("simulate.dgp.binomial_fixture_counts", x_c=x_c, n_c=n_c, x_t=x_t, n_t=n_t)


def binomial_fixture_units(
    x_c: int, n_c: int, x_t: int, n_t: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Declared-truth substrate for an exact two-independent-binomial
    rare-event fixture: ``unit_ids``, ``group_ids`` (``"control"``/
    ``"treatment"``) and a boolean ``converted`` array with EXACTLY ``x_c``
    of the first ``n_c`` (control) units and ``x_t`` of the last ``n_t``
    (treatment) units marked converted.

    A fixed count, never a ``Binomial(n, p)`` draw, preserves exact rare-event
    edge cases such as zero, one, or all successes and both directions of a
    shifted null. ``seed`` only permutes which units are converted, never how
    many.
    """
    _validate_binomial_fixture_counts(x_c, n_c, x_t, n_t)
    rng = np.random.default_rng(seed)
    unit_ids = np.array([f"c{i:06d}" for i in range(n_c)] + [f"t{i:06d}" for i in range(n_t)])
    group_ids = np.array(["control"] * n_c + ["treatment"] * n_t)
    converted = np.zeros(n_c + n_t, dtype=bool)
    converted[rng.permutation(n_c)[:x_c]] = True
    converted[n_c + rng.permutation(n_t)[:x_t]] = True
    return unit_ids, group_ids, converted


def simulate_binomial_conversion_logs(
    x_c: int, n_c: int, x_t: int, n_t: int, *, seed: int = 0, experiment_id: str = "sim"
) -> pa.Table:
    """Deterministic exact-count raw event log for two independent Bernoulli
    arms (see :func:`binomial_fixture_units`): one exposure row per unit,
    plus a single ``"conversion"`` fact row for each converted unit.

    Reuses ``simulate_raw_logs``'s own exposure-event/post-exposure-
    timestamp builders and event schema, so this flows through the exact
    same native (definitions/warehouse), frame (``from_unit_summary``) and
    artifact (``publish_unit_day_artifact``/``from_unit_day_artifact``)
    producer paths as the rest of the DGP -- the point of an "end-to-end
    raw binary fixture" rather than a hand-built group_summary row.

    Also emits one ``"conversion"`` fact row for a never-exposed
    ``"zzz_freshness_anchor"`` unit, 10 years past the experiment start.
    A window-bound conversion metric's own freshness watermark
    (``data_as_of``) is the max observed timestamp of ITS OWN fact,
    scanned over the whole fact source regardless of enrollment; without
    an anchor, this tiny fixture's own latest real event IS that
    watermark, so every unit's declared window (whatever ``window_days``
    the caller's metric definition uses) looks perpetually unclosed and
    every unit is silently censored out of ``group_summary`` -- not a
    missing feature, the correct behavior for genuinely fresh data, but
    wrong for a fixture whose truth is already fully realized. The
    anchor carries no exposure event, so it is never enrolled and never
    contributes to any metric's group_summary rows -- only to this
    watermark.
    """
    unit_ids, group_ids, converted = binomial_fixture_units(x_c, n_c, x_t, n_t, seed=seed)
    rng = np.random.default_rng(seed + 1)
    exposures, exposure_ts = _build_exposure_events(unit_ids, group_ids, rng, experiment_id)

    n_conv = int(converted.sum())
    if n_conv == 0:
        conversion_tbl = _empty_table()
    else:
        ts_list = _post_exposure_timestamps(exposure_ts[converted], 1, rng)
        conversion_tbl = pa.table(
            {
                "unit_id": pa.array(unit_ids[converted], type=pa.string()),
                "ts": pa.array(ts_list, type=pa.timestamp("us")),
                "event": pa.array(["conversion"] * n_conv, type=pa.string()),
                "experiment_id": pa.array([""] * n_conv, type=pa.string()),
                "group_id": pa.array(group_ids[converted].tolist(), type=pa.string()),
                "value": pa.array([1.0] * n_conv, type=pa.float64()),
            },
            schema=_EVENT_SCHEMA,
        )

    anchor_tbl = pa.table(
        {
            "unit_id": pa.array(["zzz_freshness_anchor"], type=pa.string()),
            "ts": pa.array([_mk_ts(3650)], type=pa.timestamp("us")),
            "event": pa.array(["conversion"], type=pa.string()),
            "experiment_id": pa.array([""], type=pa.string()),
            "group_id": pa.array([None], type=pa.string()),
            "value": pa.array([1.0], type=pa.float64()),
        },
        schema=_EVENT_SCHEMA,
    )

    combined = pa.concat_tables([exposures, conversion_tbl, anchor_tbl])
    unit_col = combined.column("unit_id").to_numpy()
    ts_col = combined.column("ts").to_numpy()
    idx = np.lexsort((ts_col, unit_col))
    return combined.take(idx)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "simulate.dgp.scenario.true_lift_finite": "true_lift[{metric!r}] must be finite and >= -0.99; got {value!r}",
        "simulate.dgp.conversion_lift_unattainable": "conversion lift {lift} is unattainable: it requires a marginal rate of {target:.4f} >= 1 (control rate {p_control:.4f})",
        "simulate.dgp.count_rate_unrepresentable": "count rate cannot be represented by the numeric generator ({reason}). {route}",
        "simulate.dgp.unsupported_metric_supported": "Unsupported metric(s): {unknown}. Supported: {supported}",
        "simulate.dgp.unknown_metric_true": "Unknown metric '{metric_name}' in true_lift",
        "simulate.dgp.uptake_compliance_t": "uptake_compliance_t and tau_complier must be set together to generate an encouragement-design uptake + outcome DGP",
        "simulate.dgp.binomial_fixture_counts": "binomial fixture counts out of range: need 0 <= x_c <= n_c, 0 <= x_t <= n_t, n_c >= 1, n_t >= 1; got x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}",
        "simulate.dgp.switchback.retained_window": "declared_carryover_order must be less than observation_steps to retain observations; got {declared_carryover_order} and {observation_steps}",
        "simulate.dgp.switchback.extra_fields": "unsupported SwitchbackScenario fields: {fields!r}",
    },
)
_raise = raiser(_REFUSALS)


class SwitchbackScenario(CodedModel, BaseModel):
    """Synthetic fixed-horizon switchback panel controls.

    Each unit receives an independent CT or TC order in every cycle, unless
    ``shared_schedule`` is set: then one order is drawn per cycle and every
    unit in that cycle realizes it (matching ``SharedScheduleOrder``). The
    supported estimand assumes carryover has decayed before admitted steps.
    ``true_carryover_order`` names the number of post-washout observation
    steps that actually still carry treatment contamination (a finite
    history), and ``carryover_amplitude`` is that contamination's size.
    ``declared_carryover_order`` is passed to ``SwitchbackWindow.carryover_order``
    by the evaluator. Nonzero contamination remains an assumption violation
    when the true order exceeds the declared order. ``treatment_effect`` is
    the additive effect over all observation steps; evaluation targets its
    retained fraction after discarding the declared history.

    ``noise_sd`` is the per-step treatment-potential error SD; control errors
    are zero. Innovations have mean zero and variance one: N(0,1),
    (exp(N(0,1))-exp(1/2))/sqrt(exp(1)*(exp(1)-1)), or
    (Gamma(shape=2,scale=1)-2)/sqrt(2). Independently for each unit, with
    probability ``within_unit_correlation`` reuse one innovation for every
    cycle/period/step; otherwise draw independent innovations. This mixture
    preserves the exact declared marginal, with Cov(e_j,e_k)=rho*noise_sd^2
    for distinct treatment-potential observations in a unit. Units and order
    draws are independent. Treatment loading prevents paired cancellation.
    This is exchangeable mixture dependence, not a Gaussian copula.

    Noise uses a separate seed stream and consumes no draws when noise_sd=0.
    Centered unit-period trends and opposite treatment-period effects permit
    heterogeneity while preserving the mean-unit retained-window target.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_units: int = Field(ge=2)
    n_cycles: int = Field(ge=1)
    treatment_effect: float = 0.0
    washout_steps: int = Field(default=1, ge=0)
    observation_steps: int = Field(default=2, ge=1)
    probability_ct: float = Field(default=0.5, gt=0.0, lt=1.0)
    unit_effect_sd: float = Field(default=1.0, ge=0.0)
    temporal_effect: float = 0.25
    declared_carryover_order: int = Field(default=0, ge=0, strict=True)
    true_carryover_order: int = Field(default=0, ge=0)
    carryover_amplitude: float = Field(default=0.0, ge=0.0)
    noise_distribution: Literal["normal", "centered_lognormal", "centered_gamma"] = "normal"
    noise_sd: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    within_unit_correlation: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    period_effect_heterogeneity: float = Field(default=0.0, allow_inf_nan=False)
    treatment_period_effect: float = Field(default=0.0, allow_inf_nan=False)
    shared_schedule: bool = False
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def _reject_extra_fields(cls, value: object) -> object:
        if isinstance(value, Mapping):
            extras = tuple(sorted(str(key) for key in value if key not in cls.model_fields))
            if extras:
                _raise("simulate.dgp.switchback.extra_fields", fields=extras)
        return value

    @model_validator(mode="after")
    def _validate_retained_window(self) -> SwitchbackScenario:
        if self.declared_carryover_order >= self.observation_steps:
            _raise(
                "simulate.dgp.switchback.retained_window",
                declared_carryover_order=self.declared_carryover_order,
                observation_steps=self.observation_steps,
            )
        return self


def _switchback_noise(scenario: SwitchbackScenario) -> np.ndarray | None:
    """Draw treatment-potential errors independently of assignment RNG state."""
    if scenario.noise_sd == 0.0:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([scenario.seed, 1414]))
    shape = (
        scenario.n_units,
        scenario.n_cycles,
        2,
        scenario.washout_steps + scenario.observation_steps,
    )

    def innovations(size: tuple[int, ...]) -> np.ndarray:
        if scenario.noise_distribution == "normal":
            return rng.normal(size=size)
        if scenario.noise_distribution == "centered_lognormal":
            return (rng.lognormal(size=size) - math.exp(0.5)) / math.sqrt(math.e * math.expm1(1.0))
        return (rng.gamma(2.0, size=size) - 2.0) / math.sqrt(2.0)

    common = innovations((scenario.n_units, 1, 1, 1))
    independent = innovations(shape)
    reuse = rng.random((scenario.n_units, 1, 1, 1)) < scenario.within_unit_correlation
    return scenario.noise_sd * np.where(reuse, common, independent)


def simulate_switchback_panel(scenario: SwitchbackScenario) -> pa.Table:
    """Generate a seeded panel under the scenario's explicit potential-outcome law."""
    rng = np.random.default_rng(scenario.seed)
    unit_effects = rng.normal(0.0, scenario.unit_effect_sd, size=scenario.n_units)
    noise = _switchback_noise(scenario)
    rows: list[dict[str, object]] = []
    total_steps = scenario.washout_steps + scenario.observation_steps
    # Drawn upfront, before any unit's loop, so the default (non-shared) path's
    # per-unit-cycle rng.binomial draws stay in their original position and every
    # existing seeded fixture keeps reproducing byte-for-byte.
    shared_orders = (
        [bool(rng.binomial(1, scenario.probability_ct)) for _ in range(scenario.n_cycles)]
        if scenario.shared_schedule
        else None
    )
    for unit_index in range(scenario.n_units):
        unit = f"u{unit_index:06d}"
        previous_treatment: bool | None = None
        for cycle in range(scenario.n_cycles):
            ct = (
                shared_orders[cycle]
                if shared_orders is not None
                else bool(rng.binomial(1, scenario.probability_ct))
            )
            order = ("control", "treatment") if ct else ("treatment", "control")
            cycle_effect = scenario.temporal_effect * (cycle + 1)
            for period, group in enumerate(order):
                for step in range(total_steps):
                    treatment = float(group == "treatment")
                    residual = 0.0
                    if previous_treatment:
                        if step < scenario.washout_steps:
                            # Supported data carry a deterministic transient
                            # that is fully excluded by the washout window.
                            residual = 1.0 * (0.5 ** (step + 1))
                        elif step < scenario.washout_steps + scenario.true_carryover_order:
                            # Only this finite number of post-washout steps
                            # carries the true contamination; steps at or
                            # past washout_steps + true_carryover_order are
                            # clean regardless of the prior period's group.
                            residual = scenario.carryover_amplitude
                    outcome = (
                        unit_effects[unit_index]
                        + cycle_effect
                        + scenario.temporal_effect * period
                        + treatment * (scenario.treatment_effect / scenario.observation_steps)
                        + residual
                    )
                    if scenario.period_effect_heterogeneity:
                        centered_unit = 2 * unit_index / (scenario.n_units - 1) - 1
                        outcome += scenario.period_effect_heterogeneity * centered_unit * period
                    if scenario.treatment_period_effect:
                        outcome += (
                            treatment
                            * scenario.treatment_period_effect
                            * (2 * period - 1)
                            / scenario.observation_steps
                        )
                    if noise is not None and treatment:
                        outcome += noise[unit_index, cycle, period, step]
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "outcome": float(outcome),
                        }
                    )
                previous_treatment = group == "treatment"
    return pa.Table.from_pylist(rows)
