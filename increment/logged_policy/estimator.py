"""Trajectory-level self-normalized IPW policy-value contrast with unit-clustered inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import t as t_dist

from increment.compatibility import IndependentUnitFloor
from increment.errors import CodedModel
from increment.estimation._tails import student_t_isf, two_sided_critical_value, wald_bounds
from increment.logged_policy._refusals import _raise
from increment.logged_policy.policy import StochasticPolicy, _validate_distribution
from increment.logged_policy.trace import LoggedTrace, _reported_probabilities

# prose: allow-long measured calibration evidence justifies the ESS threshold
# Measured on fixed-logger traces (tests/calibration/test_logged_policy_ess_floor.py
# design; 26 cells over target divergence, horizon and unit count, 9188 emitted
# intervals): coverage of the 95% interval by band of min_t ESS was 0.862 [2,5),
# 0.892 [5,10), 0.933 [10,20), 0.933 [20,40), 0.943 [40,80), 0.947 [80,160),
# 0.949 [160+), with 700-2190 intervals per band. [20,40) is the smallest band
# from which every band stays within 3 MCSE of nominal, so the floor is 20.
ESS_FLOOR = 20.0
# ESS_t <= n_units, so fewer units than the ESS floor could never pass it; the
# unit floor names that shortfall before the ESS gate can misattribute it.
INDEPENDENT_UNIT_FLOOR = IndependentUnitFloor(minimum_total_units=int(ESS_FLOOR))
# exp overflows float64 above this log-weight; the reported maximum weight becomes inf.
_LOG_FLOAT_MAX = math.log(np.finfo(np.float64).max)


class PolicyValueContrast(CodedModel, BaseModel):
    """Fixed-horizon contrast ``Delta_T = V_T(target) - V_T(reference)`` on the reward scale.

    ``ess_by_time`` is the smaller of the two policies' effective sample
    sizes at each decision index and ``max_weight_by_time`` the larger of
    their maximum cumulative weights; the per-policy tuples carry both.
    The interval is a two-sided Wald interval from the unit-clustered
    sandwich standard error with a ``t`` reference on ``n_units - 1``
    degrees of freedom.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    se: float
    lb: float
    ub: float
    alpha: float
    p_value: float
    horizon: int
    n_units: int
    ess_by_time: tuple[float, ...]
    min_propensity_by_time: tuple[float, ...]
    max_weight_by_time: tuple[float, ...]
    target_id: str
    reference_id: str
    target_version: str
    reference_version: str
    target_value: float
    reference_value: float
    target_value_by_time: tuple[float, ...]
    reference_value_by_time: tuple[float, ...]
    target_ess_by_time: tuple[float, ...]
    reference_ess_by_time: tuple[float, ...]
    target_max_weight_by_time: tuple[float, ...]
    reference_max_weight_by_time: tuple[float, ...]
    min_policy_probability_by_time: tuple[float, ...]
    logging_policy_versions: tuple[str, ...]
    update_batches: tuple[str, ...]
    reference_df: int
    null: float = 0.0
    estimand: Literal["delta_T"] = "delta_T"
    estimator: Literal["trajectory_hajek_ipw"] = "trajectory_hajek_ipw"
    reference_kind: Literal["t"] = "t"
    minimum_total_units: int = Field(default=INDEPENDENT_UNIT_FLOOR.minimum_total_units)
    ess_floor: float = ESS_FLOOR


@dataclass(frozen=True, slots=True)
class _Panel:
    """Trace laid out as ``n_units x horizon`` arrays in ``(unit_id, decision_index)`` order.

    ``centered`` holds ``Y_it - c_t`` with ``c_t`` the midpoint of the rewards
    at decision index ``t`` (``min/2 + max/2``), so every centered reward is
    bounded by half the reward range and a weighted mean of them can neither
    overflow nor round away a contrast sitting on a large common offset.
    """

    centers: np.ndarray  # horizon
    centered: np.ndarray  # n_units x horizon
    log_propensity: np.ndarray
    min_propensity_by_time: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _PolicyValues:
    """One policy's per-period centered values and influence, paired with the
    other policy's period by period in :func:`_mean_contrast_over_time`."""

    delta_by_time: np.ndarray  # V_t - c_t, |.| <= range_t / 2
    psi: np.ndarray  # n_units x horizon influence, scaled by 1/n, |.| <= range_t / 4
    value_by_time: tuple[float, ...]
    ess_by_time: tuple[float, ...]
    max_weight_by_time: tuple[float, ...]
    min_probability_by_time: tuple[float, ...]


def _panel(trace: LoggedTrace) -> _Panel:
    """Admitted records are sorted by ``(unit_id, decision_index)`` with every
    reward closed, so a plain reshape lays them out unit-major."""
    shape = (trace.n_units, trace.horizon)
    rewards = np.array([r.reward for r in trace.records], dtype=float).reshape(shape)
    propensity = np.array([r.propensity for r in trace.records], dtype=float).reshape(shape)
    centers = rewards.min(axis=0) / 2.0 + rewards.max(axis=0) / 2.0
    return _Panel(
        centers=centers,
        centered=rewards - centers,
        log_propensity=np.log(propensity),
        min_propensity_by_time=tuple(float(v) for v in propensity.min(axis=0)),
    )


def _support_gate(trace: LoggedTrace, policy: StochasticPolicy) -> np.ndarray:
    """Return ``n_units x horizon`` evaluated-policy probabilities of each chosen action.

    Refuses when the policy is not a distribution over the candidate set at
    some history, or puts positive mass on an action the logging policy
    could never choose there.
    """
    n, horizon = trace.n_units, trace.horizon
    chosen = np.empty((n, horizon))
    actions = trace.candidate_actions
    for k, (record, logged) in enumerate(
        zip(trace.records, trace.logging_distributions, strict=True)
    ):
        context = record.pre_decision_context
        reported = {a: policy.probability(a, context) for a in actions}
        probabilities = _reported_probabilities(reported)
        if probabilities is None:
            _raise(
                "logged_policy.support.policy_not_normalized",
                policy_id=policy.policy_id,
                version=policy.version,
                context=dict(context),
                probabilities=reported,
            )
        _validate_distribution(
            probabilities,
            policy_id=policy.policy_id,
            version=policy.version,
            context=dict(context),
        )
        for action, p_log in zip(actions, logged, strict=True):
            if probabilities[action] > 0.0 and p_log <= 0.0:
                _raise(
                    "logged_policy.support.action_unsupported",
                    unit_id=record.unit_id,
                    decision_index=record.decision_index,
                    policy_id=policy.policy_id,
                    version=policy.version,
                    action=action,
                    policy_probability=probabilities[action],
                )
        i, t = divmod(k, horizon)
        chosen[i, t] = probabilities[record.chosen_action]
    return chosen


def _policy_values(panel: _Panel, chosen_probability: np.ndarray) -> _PolicyValues:
    """Per-time Hajek values, diagnostics, and influence contributions for one policy.

    Cumulative log-ratios ``log W_it = sum_{s<=t} log pi(A_is|H_is) - log p_is``
    are normalized per decision time by their maximum before exponentiating, so
    the self-normalized value, ESS, and influence contributions are exact even
    when the raw product overflows or underflows. The weights are normalized
    to sum to one before they meet the centered rewards, so ``V_t - c_t`` is a
    convex combination whose partial sums stay inside the centered range.
    """
    n, horizon = panel.centered.shape
    with np.errstate(divide="ignore"):
        log_ratio = np.log(chosen_probability) - panel.log_propensity
    log_w = np.cumsum(log_ratio, axis=1)
    delta = np.empty(horizon)
    ess, max_weight = [], []
    psi = np.zeros((n, horizon))
    for t in range(horizon):
        column = log_w[:, t]
        top = float(column.max())
        if not math.isfinite(top):
            # Every unit's cumulative weight is zero: the policy never plays the logged actions.
            delta[t] = math.nan
            ess.append(0.0)
            max_weight.append(0.0)
            continue
        w = np.exp(column - top)
        total = float(w.sum())
        u = w / total
        d = panel.centered[:, t]
        delta[t] = (u * d).sum()
        ess.append(total * total / float((w * w).sum()))
        max_weight.append(math.exp(top) if top <= _LOG_FLOAT_MAX else math.inf)
        # ``u d - u V`` rather than ``u (d - V)``: a zero-weight unit contributes an
        # exact zero even where the reward range itself is beyond the float range.
        # Each product is within the range and |u (d - V)| <= range / 4, so the
        # difference cannot overflow.
        psi[:, t] = u * d - u * delta[t]
    return _PolicyValues(
        delta_by_time=delta,
        psi=psi,
        value_by_time=tuple(float(v) for v in panel.centers + delta),
        ess_by_time=tuple(ess),
        max_weight_by_time=tuple(max_weight),
        min_probability_by_time=tuple(float(v) for v in chosen_probability.min(axis=0)),
    )


def _exact_mean_difference(target: list[float], reference: list[float]) -> float:
    """``(sum(target) - sum(reference)) / len(target)`` as a correctly rounded float.

    Every finite float is ``n / 2**k``. The signed numerators are accumulated
    as Python integers at a common exponent, so no term is rounded, flushed
    to zero, or overflowed before the single integer division, which is
    correctly rounded down into the subnormal range. A quotient beyond the
    float range is returned as a signed infinity for the caller's
    representability gate.
    """
    numerator = 0
    exponent = 0
    for values, sign in ((target, 1), (reference, -1)):
        for value in values:
            n, d = value.as_integer_ratio()
            k = d.bit_length() - 1
            if k > exponent:
                numerator <<= k - exponent
                exponent = k
            numerator += sign * (n << (exponent - k))
    try:
        return numerator / (len(target) << exponent)
    except OverflowError:
        return math.inf if numerator > 0 else -math.inf


def _mean_contrast_over_time(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(1/T) sum_t (target_t - reference_t)`` over the last axis, each row
    exactly accumulated and rounded once by :func:`_exact_mean_difference`."""
    horizon = target.shape[-1]
    rows = zip(
        target.reshape(-1, horizon).tolist(),
        reference.reshape(-1, horizon).tolist(),
        strict=True,
    )
    return np.array([_exact_mean_difference(t, r) for t, r in rows]).reshape(target.shape[:-1])


def _ess_gate(
    target: StochasticPolicy,
    reference: StochasticPolicy,
    target_values: _PolicyValues,
    reference_values: _PolicyValues,
) -> None:
    for t in range(len(target_values.ess_by_time)):
        for policy, values in ((target, target_values), (reference, reference_values)):
            if values.ess_by_time[t] < ESS_FLOOR:
                _raise(
                    "logged_policy.support.ess_floor",
                    policy_id=policy.policy_id,
                    version=policy.version,
                    decision_index=t + 1,
                    ess=values.ess_by_time[t],
                    floor=ESS_FLOOR,
                    target_ess_by_time=target_values.ess_by_time,
                    reference_ess_by_time=reference_values.ess_by_time,
                )


def _sandwich_se(phi: np.ndarray) -> float:
    """``sqrt(n/(n-1) * sum_i phi_i^2)`` for the ``1/n``-scaled unit contributions.

    The squares are taken of ``phi / max|phi|`` so a contribution near the
    float range squares to at most one instead of overflowing.
    """
    n = phi.size
    scale = float(np.abs(phi).max())
    if scale == 0.0:
        return 0.0
    scaled = phi / scale
    return scale * math.sqrt(n / (n - 1) * float((scaled * scaled).sum()))


def estimate_policy_contrast(
    trace: LoggedTrace,
    target: StochasticPolicy,
    reference: StochasticPolicy,
    *,
    alpha: float = 0.05,
) -> PolicyValueContrast:
    """Estimate ``Delta_T(target, reference)`` from an admitted trace.

    Gates run in order: ``alpha``; one fixed logging law for the whole
    trace; policy support at every logged history; the independent-unit
    inference floor; the effective-sample-size floor at every decision index
    for both policies; a finite positive ``t`` critical value; a float64
    representation of every reported statistic. Each refusal is a coded
    error whose context carries the diagnostics computed so far.
    """
    if not (isinstance(alpha, float) and math.isfinite(alpha) and 0.0 < alpha < 1.0):
        _raise("logged_policy.inference.alpha", alpha=alpha)
    logging_policy_keys = trace._logging_policy_keys
    logging_policy_versions = tuple(
        f"{policy_id}/{version}" for policy_id, version in logging_policy_keys
    )
    if len(logging_policy_keys) > 1:
        _raise(
            "logged_policy.inference.adaptive_logging_unsupported",
            logging_policy_versions=logging_policy_versions,
            logging_policy_keys=logging_policy_keys,
            update_batches=trace.update_batches,
        )
    panel = _panel(trace)
    target_chosen = _support_gate(trace, target)
    reference_chosen = _support_gate(trace, reference)
    target_values = _policy_values(panel, target_chosen)
    reference_values = _policy_values(panel, reference_chosen)
    ess_by_time = tuple(map(min, target_values.ess_by_time, reference_values.ess_by_time))
    max_weight_by_time = tuple(
        map(max, target_values.max_weight_by_time, reference_values.max_weight_by_time)
    )
    n, horizon = panel.centered.shape
    if n < INDEPENDENT_UNIT_FLOOR.minimum_total_units:
        _raise(
            "logged_policy.inference.cluster_floor",
            n_units=n,
            minimum_total_units=INDEPENDENT_UNIT_FLOOR.minimum_total_units,
            ess_by_time=ess_by_time,
            max_weight_by_time=max_weight_by_time,
            min_propensity_by_time=panel.min_propensity_by_time,
        )
    _ess_gate(target, reference, target_values, reference_values)
    df = n - 1
    q = two_sided_critical_value(student_t_isf, alpha, df, what="logged-policy contrast interval")
    # Pair each policy's influence period by period. Means and drift correction
    # scale by 1/T or 1/n before summing; the shared Wald helper handles
    # non-finite values and interval bounds.
    phi = _mean_contrast_over_time(target_values.psi, reference_values.psi)
    phi = phi - (phi / n).sum()  # exactly zero-mean in exact arithmetic; removes float drift
    se = _sandwich_se(phi)
    estimate = float(
        _mean_contrast_over_time(target_values.delta_by_time, reference_values.delta_by_time)
    )
    target_value = float((np.asarray(target_values.value_by_time) / horizon).sum())
    reference_value = float((np.asarray(reference_values.value_by_time) / horizon).sum())
    reported = {
        "estimate": estimate,
        "se": se,
        "target_value": target_value,
        "reference_value": reference_value,
    }
    nonfinite = tuple(name for name, value in reported.items() if not math.isfinite(value))
    if nonfinite:
        _raise("logged_policy.inference.unrepresentable", nonfinite=nonfinite, **reported)
    lb, ub = wald_bounds(estimate, q, se, what="logged-policy contrast interval")
    p_value = (
        float(2.0 * t_dist.sf(abs(estimate) / se, df))
        if se > 0.0
        else (1.0 if estimate == 0.0 else 0.0)
    )
    return PolicyValueContrast(
        estimate=estimate,
        se=se,
        lb=lb,
        ub=ub,
        alpha=alpha,
        p_value=p_value,
        horizon=trace.horizon,
        n_units=n,
        ess_by_time=ess_by_time,
        min_propensity_by_time=panel.min_propensity_by_time,
        max_weight_by_time=max_weight_by_time,
        target_id=target.policy_id,
        reference_id=reference.policy_id,
        target_version=target.version,
        reference_version=reference.version,
        target_value=target_value,
        reference_value=reference_value,
        target_value_by_time=target_values.value_by_time,
        reference_value_by_time=reference_values.value_by_time,
        target_ess_by_time=target_values.ess_by_time,
        reference_ess_by_time=reference_values.ess_by_time,
        target_max_weight_by_time=target_values.max_weight_by_time,
        reference_max_weight_by_time=reference_values.max_weight_by_time,
        min_policy_probability_by_time=tuple(
            map(
                min, target_values.min_probability_by_time, reference_values.min_probability_by_time
            )
        ),
        logging_policy_versions=trace.logging_policy_versions,
        update_batches=trace.update_batches,
        reference_df=df,
    )
