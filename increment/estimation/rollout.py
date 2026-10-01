"""Segment rollout recommendation over declared discrete dimensions.

Given per-segment effect moments ``(est_k, var_k)`` - the same contract
:func:`increment.estimation.meta.marginalized_segment_intervals`
consumes - recommends which segments clear a rollout cost threshold
and reports the recommendation's policy value with the winner's curse
removed.

Selection is exact and cheap: the policy value ``V(S) = sum_{k in S}
(theta_k - c)`` is additive over disjoint segments, so the
value-maximising subset is precisely ``{k : est_k > c}``.

Picking the subset with the best estimated value biases that estimate
upward (selection / winner's-curse bias); the reported value subtracts
a closed-form correction, the posterior mean of the exact
selection-bias functional
``E[(est_k - theta_k) 1{est_k > c}] = s_k phi((c - theta_k)/s_k)``
under the same tau-marginalised posterior the shrinkage integrates
over. Verified against a pre-stated bar (corrected bias
<= 25% of the raw selection bias across a K x spread simulation grid,
20k reps/cell): shrunken-only leaves ~half the bias (worst ratio
0.551), and a truncated-normal conditional MLE is worse than no
correction at all (worst ratio 2.041, unbounded below).

The correction degrades when the cost threshold sits far above the
typical segment effect (selection is almost pure noise); that regime
is detectable from the same posterior and refused rather than
shipped: when the offset ``(c - mu_post) / s_rms`` reaches
``_OFFSET_GUARD_THRESHOLD``, the value fields come back ``None`` with
``recommendation="refuse"`` - a result, not an exception (same
contract as :class:`increment.estimation.targeting.TargetingRule`).

Inside the guard's regime the corrected value can still come back
<= 0: when segment noise dominates the effects, the correction can
equal or exceed the raw value, and ``recommendation="no_net_benefit"``
reports that with every value field still populated. A negative
``policy_value`` is not a prediction of losses - it is the
procedure-level debiased number, not the posterior value of the
realized subset, and the two can disagree in sign.
``selection_bias_share`` makes this legible: at >= 1 the correction
consumed the entire apparent lift.

``est``/``var`` are calibrated for log-scale relative effects, sharing
a prior with the shrinkage - see the ``tau_prior_scale`` warnings on
:func:`marginalized_segment_intervals` before passing anything else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.estimation.meta import (
    _TAU_PRIOR_SCALE_DEFAULT,
    _tau_posterior,
    _validate_est_var,
    _validate_tau_prior_scale,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.rollout.segment_rollout.selected_bool_length": "selected must be 1-D bool with length k={k}, got shape {shape} dtype {dtype}",
        "estimation.rollout.segment_rollout.policy_value_policy": "policy_value, policy_value_raw and selection_bias must be set together or None together",
        "estimation.rollout.segment_rollout.recommendation_refuse_carry": "recommendation='refuse' must carry None value fields, and 'rollout'/'no_net_benefit' must carry populated ones — they travel together",
        "estimation.rollout.segment_rollout.recommendation_match_value": "recommendation must match the value's sign: 'rollout' iff policy_value > 0, 'no_net_benefit' iff policy_value <= 0",
        "estimation.rollout.segment_rollout.selection_bias_share": "selection_bias_share must be None exactly behind a refusal or when policy_value_raw == 0, and populated otherwise",
        "estimation.rollout.cost_threshold_finite": "cost_threshold must be finite, got {cost_threshold}",
    },
)
_raise = raiser(_REFUSALS)

_OFFSET_GUARD_THRESHOLD = 2.0
"""Refuse to report a value when ``(c - mu_post)/s_rms`` reaches this.

Measured: the largest threshold satisfying two pre-stated conditions
on a 20,000-rep x 45-cell simulation grid - corrected bias <= 25% of
the raw selection bias at every in-scope cell (worst 0.237 +/- 0.004),
while refusing at most 2.3% of the healthy (offset <= 1) regime
against a 5% bar. Lower thresholds (1.0, 1.5) refuse 13-40% of the
healthy regime; higher ones (2.5, 3.0) let the bias ratio reach
0.425-0.470.

``s_rms`` is ``sqrt(mean(var_k))``, and the threshold is calibrated in
exactly those units - any other normalisation silently voids this
measurement. A module constant rather than a parameter deliberately:
a tunable guard is no guard.
"""


class SegmentRollout(CodedModel, BaseModel):
    """Which segments to roll out, and an honest price on doing so.

    ``policy_value``, ``policy_value_raw`` and ``selection_bias`` are
    populated together and ``None`` together. ``recommendation`` carries
    two claims: the subset (``selected``, sound at any precision - an
    argmax) and the price. ``"rollout"`` asserts the honest value is
    positive; ``"no_net_benefit"`` keeps the subset and full accounting
    but says the evidence cannot demonstrate positive value;
    ``"refuse"`` (offset guard) withholds the value fields entirely.
    ``selected`` and ``estimated_offset`` are attached in every state,
    but a ``"refuse"`` recommendation means the selection is
    noise-dominated and must not be deployed on this evidence alone.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    __hash__ = None
    """Unhashable, same rationale as ``MarginalizedSegmentIntervals``."""

    k: int
    cost_threshold: float
    """Per-segment break-even effect ``c``, on the scale of ``est``."""
    selected: np.ndarray
    """(k,) bool: ``est_k > cost_threshold``, the exact value-maximising subset."""
    estimated_offset: float
    """``(c - mu_post) / s_rms``: how many typical SEs the posterior pooled
    mean sits below the bar.  The guard statistic."""
    recommendation: Literal["rollout", "no_net_benefit", "refuse"]
    policy_value: float | None
    """Selection-bias-corrected total lift of rolling out ``selected``:
    ``policy_value_raw - selection_bias``. Point estimate only - no
    interval ships because only the bias, not any coverage, was validated."""
    policy_value_raw: float | None
    """Uncorrected ``sum(est_k - c)`` over the selected segments."""
    selection_bias: float | None
    """Posterior-mean winner's-curse bias of ``policy_value_raw``; >= 0."""
    selection_bias_share: float | None
    """``selection_bias / policy_value_raw``: the share of the raw value
    the correction consumed. ``>= 1`` means the winner's curse accounts
    for the entire apparent lift. ``None`` behind a refusal, and when
    ``policy_value_raw == 0`` (empty selection), where the ratio is undefined."""

    @model_validator(mode="after")
    def _check_consistency(self) -> SegmentRollout:
        sel = self.selected
        if sel.ndim != 1 or sel.shape[0] != self.k or sel.dtype != np.bool_:
            _raise(
                "estimation.rollout.segment_rollout.selected_bool_length",
                k=self.k,
                shape=sel.shape,
                dtype=str(sel.dtype),
            )
        values = (self.policy_value, self.policy_value_raw, self.selection_bias)
        if any(v is None for v in values) != all(v is None for v in values):
            _raise("estimation.rollout.segment_rollout.policy_value_policy")
        if (self.recommendation == "refuse") != (self.policy_value is None):
            _raise("estimation.rollout.segment_rollout.recommendation_refuse_carry")
        if self.policy_value is not None and (self.policy_value > 0.0) != (
            self.recommendation == "rollout"
        ):
            _raise("estimation.rollout.segment_rollout.recommendation_match_value")
        if (self.selection_bias_share is None) != (
            self.selection_bias is None or self.policy_value_raw == 0.0
        ):
            _raise("estimation.rollout.segment_rollout.selection_bias_share")
        return self


def segment_rollout(
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    *,
    cost_threshold: float = 0.0,
    tau_prior_scale: float = _TAU_PRIOR_SCALE_DEFAULT,
) -> SegmentRollout:
    """Recommend a rollout subset and honestly price it. See the module
    docstring for the estimator, its evidence, and the refusal contract.

    ``est``/``var`` are per-segment point estimates and sampling
    variances (log-scale relative effects; same contract as
    :func:`marginalized_segment_intervals`). ``cost_threshold`` is the
    break-even per-segment effect ``c`` - a segment belongs in the
    rollout iff its true effect exceeds this, same scale as ``est``.
    ``tau_prior_scale`` is the HalfNormal prior scale on tau (internal
    default; see the coverage caveats on
    :func:`marginalized_segment_intervals` before changing).

    Raises ``ValueError`` on the same input guards as
    ``marginalized_segment_intervals``, plus a non-finite
    ``cost_threshold``. A posterior that cannot be integrated to the
    required accuracy within the numerical budget raises
    ``estimation.meta.posterior_integration_unresolved``: it is a
    numerical failure, never reported as the offset guard's ``"refuse"``.
    """
    est_arr, var_arr = _validate_est_var(est, var)
    _validate_tau_prior_scale(tau_prior_scale)
    if not np.isfinite(cost_threshold):
        _raise("estimation.rollout.cost_threshold_finite", cost_threshold=cost_threshold)

    c = float(cost_threshold)
    k = est_arr.shape[0]
    posterior = _tau_posterior(est_arr, var_arr, tau_prior_scale, predictive_at=c)

    selected = est_arr > c
    mu_post = posterior.pooled_mean
    s_rms = float(np.sqrt(var_arr.mean()))
    estimated_offset = (c - mu_post) / s_rms

    if estimated_offset >= _OFFSET_GUARD_THRESHOLD:
        return SegmentRollout(
            k=k,
            cost_threshold=c,
            selected=selected,
            estimated_offset=estimated_offset,
            recommendation="refuse",
            policy_value=None,
            policy_value_raw=None,
            selection_bias=None,
            selection_bias_share=None,
        )

    raw_value = float((est_arr - c)[selected].sum())
    # Posterior mean of the selection-bias functional, closed form: with
    # theta_k | y, tau ~ N(cond_mean, cond_var), E[s_k phi((c - theta_k)/s_k)]
    # = v_k N(c; cond_mean_k, v_k + cond_var_k), whose tau-average is v_k
    # times the posterior predictive density of a replicate estimate at c.
    assert posterior.predictive_density is not None
    selection_bias = float(var_arr @ posterior.predictive_density)

    value = raw_value - selection_bias
    return SegmentRollout(
        k=k,
        cost_threshold=c,
        selected=selected,
        estimated_offset=estimated_offset,
        recommendation="rollout" if value > 0.0 else "no_net_benefit",
        policy_value=value,
        policy_value_raw=raw_value,
        selection_bias=selection_bias,
        selection_bias_share=(None if raw_value == 0.0 else selection_bias / raw_value),
    )
