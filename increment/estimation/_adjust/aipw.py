"""AIPW observational adjustment estimator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from increment._literals import PreferredDirection, ValueScale
from increment.estimation._adjust.common import (
    AdjustedContrastRequest,
    AdjustmentCohort,
    AdjustmentData,
    ClusterSupport,
    NuisanceFoldSpec,
    OutcomeRole,
    _apply_overlap_gate,
    _cohort_clusters,
    _crossfit_nuisances,
    _infer_adjusted_contrast,
    _joint_reference_from_influences,
    _prepare_adjustment_cohort,
    _prepare_adjustment_requests,
    _refuse_near_zero_adjustment_denominator,
    _resolve_adjustment_missingness,
    _score_stats,
    _validate_contrast,
)
from increment.estimation._adjust.learners import Learner, LogisticPropensity, RidgeOutcome
from increment.estimation._adjust.value_scale import resolve_value_scale
from increment.estimation._adjust.weight_diagnostics import _weight_summary
from increment.estimation.adjust import ADJUSTMENTS
from increment.estimation.inference import Prior

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


def _augmented_arm_mean(
    m_hat: np.ndarray, weight: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Normalized augmented mean of one arm over the cohort and its score.

    With w = I_a / p_a, W = sum(w), residual r = y - m and b = sum(w r) / W,
    mu_a = mean(m) + b and psi_a = m + (N w / W)(r - b) + b, so psi_a - mu_a
    is arm a's influence: the outcome-prediction population term plus the
    self-normalized residual correction. At m == 0 it is the Hajek mean.
    """
    n = y.shape[0]
    big_w = float(weight.sum())
    r = y - m_hat
    b = float((weight * r).sum() / big_w)
    mu = float(m_hat.mean() + b)
    psi = m_hat + (n * weight / big_w) * (r - b) + b
    return mu, psi


def _aipw_contrast(
    request: AdjustedContrastRequest,
    *,
    data: AdjustmentData,
    clusters: ClusterSupport,
    mu1: float,
    mu0: float,
    psi1: np.ndarray,
    psi0: np.ndarray,
) -> LiftEstimate:
    """Infer one treatment's contrast from its arm and the shared control's
    augmented means and aligned cohort scores, reduced over every cohort
    cluster (`clusters`): the population terms reach every cohort row."""
    tau = mu1 - mu0
    if_tau = psi1 - psi0
    if_tau_centered = if_tau - float(if_tau.mean())
    abs_scores = _score_stats(
        if_tau_centered,
        metric=request.metric.name,
        contrast=request.treatment_group,
        support=clusters,
    )
    resolved = resolve_value_scale(
        value_scale=request.value_scale,
        prior=request.prior,
        raw_point=float(tau),
        raw_scores=abs_scores,
        mu0=mu0,
        mu0_se=lambda: _score_stats(
            psi0 - mu0,
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=clusters,
        ).se(),
        lift_fn=lambda mu0_value: tau / mu0_value,
        lift_scores_fn=lambda mu0_value: _score_stats(
            (psi1 - (1.0 + tau / mu0_value) * psi0) / mu0_value,
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=clusters,
        ),
        refuse_near_zero=lambda mu0_value, se: _refuse_near_zero_adjustment_denominator(
            request, mu0_value, se, label="AIPW control mean"
        ),
    )
    point, scores = resolved.point, resolved.scores
    abs_diff, abs_se = resolved.abs_diff, resolved.abs_se
    if request.value_scale != "absolute":
        joint_result = _joint_reference_from_influences(
            if_tau_centered,
            psi0 - mu0,
            numerator_point=float(tau),
            denominator_point=float(mu0),
            supports=(clusters, clusters),
        )
        if joint_result[1] is not None and mu0 != 0.0:
            ratio_point = float(tau) / float(mu0)
            point = ratio_point if np.isfinite(ratio_point) else None
    else:
        joint_result = (None, None)
    return _infer_adjusted_contrast(
        request,
        point=point,
        scores=scores,
        population=data.population,
        absolute=(abs_diff, abs_se),
        estimand="overlap_subpopulation_ate" if data.overlap_trimmed else "ate",
        note=data.note,
        n_clusters=clusters.k,
        dof=None,
        posterior=(resolved.posterior_point, resolved.posterior_scores),
        joint_result=joint_result,
    )


def _aipw_cohort(
    cohort: AdjustmentCohort, spec: NuisanceFoldSpec, folds: int
) -> list[LiftEstimate]:
    """Estimate every treatment's AIPW contrast over one shared cohort."""
    data = _prepare_adjustment_cohort(cohort)
    data = _resolve_adjustment_missingness(cohort, data)
    nuisance = _crossfit_nuisances(cohort, data, spec, folds)
    data, nuisance = _apply_overlap_gate(cohort, data, nuisance)
    clusters = _cohort_clusters(data)
    p = nuisance.marginal
    arms = range(data.n_treatments + 1)
    weights = [data.indicator(t) / p[:, t] for t in arms]
    for a in range(1, data.n_treatments + 1):
        _validate_contrast(cohort, data, a, weights[a] + weights[0], clusters)
    means = [
        _augmented_arm_mean(nuisance.outcomes[OutcomeRole("arm", t)], weights[t], data.y)
        for t in arms
    ]
    mu0, psi0 = means[0]
    summary = [
        _weight_summary(w, clusters.inv, clusters.k)
        if clusters.inv is not None
        else _weight_summary(w)
        for w in weights
    ]
    definition = (
        "unclipped inverse marginal propensity I(A=a)/p_a(X), retained cohort; "
        "AIPW self-normalized residual correction"
    )
    grain = "cluster" if clusters.inv is not None else "unit"
    return [
        _aipw_contrast(
            request,
            data=data,
            clusters=clusters,
            mu1=means[a][0],
            mu0=mu0,
            psi1=means[a][1],
            psi0=psi0,
        ).model_copy(
            update={
                "weight_diagnostics_available": True,
                "weight_diagnostics_reason_code": None,
                "weight_diagnostics_reason_context": None,
                "weight_definition": definition,
                "weight_grain": grain,
                "control_weight_n": summary[0][0],
                "treatment_weight_n": summary[a][0],
                "control_weight_ess": summary[0][1],
                "treatment_weight_ess": summary[a][1],
                "control_weight_max_share": summary[0][2],
                "treatment_weight_max_share": summary[a][2],
            }
        )
        for a, request in enumerate(cohort.requests, start=1)
    ]


# Public estimator signature is the API for AIPW results.
def aipw_estimate(  # noqa: PLR0913
    src: MomentSource,
    metric: Metric,
    design: Observational,
    *,
    propensity_learner: Callable[[], Learner] | None = None,
    outcome_learner: Callable[[], Learner] | None = None,
    folds: int = 5,
    prior: Prior | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    null_lift: float = 0.0,
    null_abs: float | None = None,
    value_scale: ValueScale = "relative",
    preferred_direction: PreferredDirection | None = None,
    moment_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[LiftEstimate]:
    """AIPW (augmented inverse-propensity-weighted, doubly-robust) estimate
    of unit-weighted ATE-scale relative lift, one entry per non-control
    treatment arm present for *metric*.

    Cross-fits one propensity model per treatment (on that treatment and
    control, coupled into marginal arm propensities over the whole cohort)
    and one outcome model per arm (unlike `dml_estimate`'s pooled
    partially-linear outcome model), then Hajek-stabilizes the IPW
    correction term. Every arm mean averages its outcome predictions over
    the same eligible cohort, so all treatments share one control mean.
    Registered under `Method.name == "aipw"`.

    Unlike `"dml"` (a partially-linear PLR estimand, propensity-variance-
    weighted; see the estimand note in `dml.py`), `"aipw"` targets
    the population ATE proper on the unit-weighted scale, matching
    `"iptw"`'s estimand exactly but with the added protection of a correct
    outcome model when the propensity model is wrong (double robustness).

    `propensity_learner`/`outcome_learner` are factories (zero-arg
    callables returning a fresh `Learner`): a fresh instance is fit per
    fold per model, matching `dml_estimate`. Default to
    `LogisticPropensity` and `RidgeOutcome` respectively.

    A cluster declared on the source (`src.cluster`) switches the SE to
    cluster totals of the influence function with an asymptotic Normal reference,
    the same seam and small-K policy as `iptw_estimate`.
    """
    requests, learners = _prepare_adjustment_requests(
        src,
        metric,
        design,
        method="AIPW",
        covariates=list(design.adjustment.covariates),
        learner_roles=("propensity_learner", "outcome_learner"),
        propensity_learner=propensity_learner,
        outcome_learner=outcome_learner,
        prior=prior,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        value_scale=value_scale,
        preferred_direction=preferred_direction,
        moment_rows=moment_rows,
    )
    if not requests:
        return []
    treatments = range(1, len(requests) + 1)
    spec = NuisanceFoldSpec(
        propensity_factory=learners["propensity_learner"] or LogisticPropensity,
        outcome_factory=learners["outcome_learner"] or RidgeOutcome,
        # Treatment arms before the shared control arm, as in the binary schedule.
        outcome_roles=(*(OutcomeRole("arm", a) for a in treatments), OutcomeRole("arm", 0)),
        deferred_roles=(),
    )
    return _aipw_cohort(AdjustmentCohort(tuple(requests)), spec, folds)


ADJUSTMENTS.register("aipw", aipw_estimate)
