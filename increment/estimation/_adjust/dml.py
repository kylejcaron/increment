"""DML observational adjustment estimator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from increment._literals import PreferredDirection, ValueScale
from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation._adjust.common import (
    AdjustedContrastRequest,
    AdjustmentCohort,
    AdjustmentData,
    ClusterSupport,
    NuisanceFoldSpec,
    NuisancePredictions,
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
from increment.estimation.adjust import ADJUSTMENTS
from increment.estimation.armstats import ScoreStats
from increment.estimation.inference import Prior

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.adjust_dml.dml_residualized_treatment": "DML residualized treatment has zero variation for metric {metric_name!r} (contrast {contrast!r}) -- the propensity model perfectly predicts treatment assignment on this data; theta is not identified",
    },
)
_raise = raiser(_REFUSALS)

_PLR_ESTIMAND_NOTE = (
    "DML reports the partially-linear slope (estimand='plr_slope'): the "
    "conditional effect averaged over the cohort with weight p_a p_0 / (p_a + p_0) "
    "in the marginal arm propensities (with one treatment, the treatment variance "
    "e(1 - e)), which equals the ATE under homogeneous effects or a constant weight "
    "but not in general; a relative row divides it by the augmented control-arm mean "
    "of the whole cohort. Use 'iptw' or 'aipw' for the unweighted ATE under "
    "heterogeneity"
)


def _compose_plr_note(note: str | None) -> str:
    """Prepend whatever caveat the row already carries (e.g. the
    impute-indicator/allow note) to the PLR-estimand caveat; every DML row
    gets the PLR caveat, never just the data-handling one.
    """
    return "; ".join(filter(None, [note, _PLR_ESTIMAND_NOTE]))


def _dml_theta(
    d_tilde: np.ndarray,
    y_tilde: np.ndarray,
    *,
    metric_name: str,
    contrast: str,
    support: ClusterSupport | None = None,
) -> tuple[float, np.ndarray, ScoreStats]:
    """Pooled DML2 partialling-out solution for one contrast.

    Given cross-fitted residuals `d_tilde = d - e_hat` (treatment minus its
    propensity) and `y_tilde = y - m_hat` (outcome minus its pooled mean),
    solves the pooled estimating equation
    `sum(d_tilde * (y_tilde - theta * d_tilde)) == 0` for `theta` in
    closed form: fold-level sums are additive, so pooling every fold's
    residuals at once is exact, not an approximation of fold-averaging.
    The caller passes one comparison's rows, and `support` indexes their
    clusters: the score vanishes outside the comparison, so it carries no
    information from a cluster holding only other treatments.

    Returns `(theta_hat, psi, scores)`. `psi` is the per-unit score at
    the solution: `psi.sum() == 0` up to float error, since this IS the
    estimating equation (mirroring the Hajek `sum_psi == 0` invariant in
    `iptw.py`). `scores.sum_d_tilde2` is the sandwich normalizer;
    `scores.se()` is `theta_hat`'s additive-scale standard error.
    """
    sum_d_tilde2 = float((d_tilde**2).sum())
    if sum_d_tilde2 <= 0:
        _raise(
            "estimation.adjust_dml.dml_residualized_treatment",
            metric_name=metric_name,
            contrast=contrast,
        )
    theta_hat = float((d_tilde * y_tilde).sum() / sum_d_tilde2)
    psi = d_tilde * (y_tilde - theta_hat * d_tilde)
    scores = _score_stats(
        psi,
        metric=metric_name,
        contrast=contrast,
        normalizer=sum_d_tilde2,
        support=support,
    )
    return theta_hat, psi, scores


def _control_mean(
    m0_hat: np.ndarray, weight: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Augmented control-arm mean over the cohort and its influence.

    With w = I_0 / p_0, W = sum(w), r = y - m0 and b = sum(w r) / W:
    mu0 = mean(m0) + b and phi0 = m0 - mean(m0) + (N w / W)(r - b). The
    control regression is fitted on control rows only; neither the pooled
    regression nor a constant-effect shift of it is a control mean under
    heterogeneous effects.
    """
    n = y.shape[0]
    big_w = float(weight.sum())
    r = y - m0_hat
    b = float((weight * r).sum() / big_w)
    mu0 = float(m0_hat.mean() + b)
    psi0 = m0_hat + (n * weight / big_w) * (r - b) + b
    return mu0, psi0 - mu0


def _dml_contrast(
    request: AdjustedContrastRequest,
    a: int,
    *,
    data: AdjustmentData,
    nuisance: NuisancePredictions,
    support: ClusterSupport,
    control: tuple[float, np.ndarray, ClusterSupport] | None,
) -> LiftEstimate:
    """Solve one treatment's pooled PLR slope on its comparison rows and
    infer it, relative to the shared augmented control mean when requested.

    The slope's score vanishes outside the comparison, so it is reduced on
    the comparison's own rows and clusters (`support`). The control mean's
    influence has population terms on every cohort row, so it, and every
    combination with it, reduces over the cohort's clusters."""
    rows = support.rows
    d_tilde = (data.arm[rows] == a).astype(float) - nuisance.conditional[rows, a - 1]
    y_tilde = data.y[rows] - nuisance.outcomes[OutcomeRole("pair", a)][rows]
    theta_hat, psi, theta_scores = _dml_theta(
        d_tilde,
        y_tilde,
        metric_name=request.metric.name,
        contrast=request.treatment_group,
        support=ClusterSupport(slice(None), support.inv, support.k),
    )
    sum_d_tilde2 = theta_scores.sum_d_tilde2
    assert sum_d_tilde2 is not None
    n = data.arm.shape[0]
    # The control mean only serves the relative scale; an absolute row never
    # fits, weights or reduces it, and never forms these influences.
    mu0 = control[0] if control is not None else 0.0

    def slope_influence() -> np.ndarray:
        """The slope's cohort-aligned influence, zero outside the comparison."""
        if_theta = np.zeros(n)
        if_theta[rows] = n * psi / sum_d_tilde2
        return if_theta

    def lift_scores(mu0_value: float) -> ScoreStats:
        assert control is not None
        return _score_stats(
            (slope_influence() - (theta_hat / mu0_value) * control[1]) / mu0_value,
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=control[2],
        )

    def mu0_se() -> float:
        assert control is not None
        return _score_stats(
            control[1],
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=control[2],
        ).se()

    resolved = resolve_value_scale(
        value_scale=request.value_scale,
        prior=request.prior,
        raw_point=theta_hat,
        raw_scores=theta_scores,
        mu0=mu0,
        mu0_se=mu0_se,
        lift_fn=lambda mu0_value: theta_hat / mu0_value,
        lift_scores_fn=lift_scores,
        refuse_near_zero=lambda mu0_value, se: _refuse_near_zero_adjustment_denominator(
            request, mu0_value, se, label="DML control mean"
        ),
    )
    point, scores = resolved.point, resolved.scores
    abs_diff, abs_se = resolved.abs_diff, resolved.abs_se
    if control is not None and request.prior is None:
        joint_result = _joint_reference_from_influences(
            slope_influence(),
            control[1],
            numerator_point=float(theta_hat),
            denominator_point=float(mu0),
            supports=(support, control[2]),
        )
        if joint_result[1] is not None and mu0 != 0.0:
            ratio_point = float(theta_hat) / float(mu0)
            point = ratio_point if np.isfinite(ratio_point) else None
    else:
        joint_result = (None, None)
    return _infer_adjusted_contrast(
        request,
        point=point,
        scores=scores,
        population=data.population,
        abs_diff=abs_diff,
        abs_se=abs_se,
        estimand="plr_slope",
        note=_compose_plr_note(data.note),
        # A relative row's control mean draws on every cohort cluster.
        n_clusters=support.k if control is None else control[2].k,
        dof=None,
        joint_result=joint_result,
    )


def _dml_cohort(cohort: AdjustmentCohort, spec: NuisanceFoldSpec, folds: int) -> list[LiftEstimate]:
    """Estimate every treatment's DML slope over one shared cohort."""
    data = _prepare_adjustment_cohort(cohort)
    data = _resolve_adjustment_missingness(cohort, data)
    nuisance = _crossfit_nuisances(cohort, data, spec, folds)
    data, nuisance = _apply_overlap_gate(cohort, data, nuisance)
    p = nuisance.marginal
    control_weight = data.indicator(0) / p[:, 0]
    clusters = _cohort_clusters(data)
    supports = [
        _validate_contrast(cohort, data, a, data.indicator(a) / p[:, a] + control_weight, clusters)
        for a in range(1, data.n_treatments + 1)
    ]
    control_role = OutcomeRole("arm", 0)
    control: tuple[float, np.ndarray, ClusterSupport] | None = None
    if control_role in nuisance.outcomes:
        mu0, phi0 = _control_mean(nuisance.outcomes[control_role], control_weight, data.y)
        control = (mu0, phi0, clusters)
    return [
        _dml_contrast(request, a, data=data, nuisance=nuisance, support=support, control=control)
        for a, (request, support) in enumerate(zip(cohort.requests, supports, strict=True), start=1)
    ]


# Public estimator signature is the API for DML results.
def dml_estimate(  # noqa: PLR0913
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
    """DML (double machine learning, partially-linear cross-fit
    partialling-out) estimate of relative lift, one entry per non-control
    treatment arm present for *metric*.

    Each treatment's slope pools the partialling-out score over its own
    treatment and control rows, with a propensity and pooled outcome model
    trained on those rows (`estimand="plr_slope"`: the effect weighted by
    the within-comparison conditional treatment variance, not the ATE in
    general). A relative row divides that slope by the augmented control
    mean of the whole eligible cohort: a control-arm outcome regression
    cross-fitted on the same folds after the slope's own fits, corrected by
    control residuals weighted by the coupled marginal control propensity.
    The slope and control mean enter one joint covariance for the relative
    confidence set. An absolute request fits no control regression.

    Cross-fitting lives in `_crossfit_nuisances`; see
    `docs/guides/observational.md` for choosing among `"iptw"`, `"dml"`,
    and `"aipw"`. Registered under `Method.name == "dml"`.

    `propensity_learner`/`outcome_learner` are factories (zero-arg
    callables returning a fresh `Learner`): a fresh instance is fit per
    fold since a `Learner` is stateful. Default to `LogisticPropensity`
    and `RidgeOutcome` respectively.

    A cluster declared on the source (`src.cluster`) switches the
    sandwich numerator to cluster totals of the score with an asymptotic Normal
    reference, the same seam and small-K policy as `iptw_estimate`. The
    slope's score vanishes outside its comparison, so it is reduced over the
    comparison's own clusters (the absolute row and the additive sidecar);
    the control mean's influence reduces over every cohort cluster, and each
    keeps its own K/(K-1) inside the joint covariance.
    """
    requests, learners = _prepare_adjustment_requests(
        src,
        metric,
        design,
        method="DML",
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
    spec = NuisanceFoldSpec(
        propensity_factory=learners["propensity_learner"] or LogisticPropensity,
        outcome_factory=learners["outcome_learner"] or RidgeOutcome,
        outcome_roles=tuple(OutcomeRole("pair", a) for a in range(1, len(requests) + 1)),
        deferred_roles=(OutcomeRole("arm", 0),) if value_scale != "absolute" else (),
    )
    return _dml_cohort(AdjustmentCohort(tuple(requests)), spec, folds)


ADJUSTMENTS.register("dml", dml_estimate)
