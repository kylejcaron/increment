"""IPTW observational adjustment estimator.

The Hájek estimator, not Horvitz-Thompson, is this package's default
IPTW psi: self-normalized weights are location-invariant and bounded
under propensity misfit, a property DML's cross-fit score generalizes.

Every arm mean is a Hájek mean over the whole eligible cohort, weighted by
the inverse of that arm's marginal propensity; each treatment's contrast
shares the one control mean and its influence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from increment._literals import PreferredDirection, ValueScale
from increment.estimation._adjust.common import (
    AdjustedContrastRequest,
    AdjustmentCohort,
    AdjustmentData,
    ClusterSupport,
    NuisancePredictions,
    _apply_overlap_gate,
    _cohort_clusters,
    _comparison_suffix,
    _couple_propensities,
    _disclose_unseen_levels,
    _fit_predict,
    _infer_adjusted_contrast,
    _joint_reference_from_influences,
    _prepare_adjustment_cohort,
    _prepare_adjustment_requests,
    _refuse_near_zero_adjustment_denominator,
    _resolve_adjustment_missingness,
    _score_stats,
    _validate_contrast,
)
from increment.estimation._adjust.encoding import (
    UnseenLevels,
    encoded_factory,
    fitted_design,
    learner_name,
    unwrap,
)
from increment.estimation._adjust.learners import (
    Learner,
    LogisticPropensity,
    _coupled_hajek_mean_influences,
)
from increment.estimation._adjust.overlap import (
    _finite_guard,
    _impute_with_indicators,
    _pattern_propensities,
    _probe_nan_capability,
    _propensity_range_guard,
)
from increment.estimation._adjust.value_scale import resolve_value_scale
from increment.estimation._adjust.weight_diagnostics import _weight_summary
from increment.estimation.adjust import ADJUSTMENTS
from increment.estimation.inference import Prior

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric
    from increment.sources import MomentSource

_FITTED_LOGISTIC_NOTE = "Joint fitted-logistic/Hajek estimating-equation covariance."
_FIXED_PROPENSITY_NOTE = (
    "Propensity predictions treated as fixed in the covariance; fitted generic, "
    "pattern-specific or trimmed propensity uncertainty is unresolved."
)


def _fit_propensities(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    factory: Callable[[], Learner],
) -> tuple[NuisancePredictions, list[Learner]]:
    """Fit IPTW's in-sample treatment-versus-control propensities, including
    the pattern strategy, predict each on every cohort row, and couple them.

    A fresh learner serves each comparison; it trains on that comparison's
    rows only, and fits its level encoding there, so another treatment's
    rows can carry a level it never saw -- disclosed once every
    comparison is fitted."""
    n = data.arm.shape[0]
    metric_name = cohort.metric.name
    control_group = cohort.control_group
    conditional = np.empty((n, data.n_treatments), order="F")
    learners: list[Learner] = []
    n_patterns = 0
    make_learner = encoded_factory(factory, data.layout)
    unseen = UnseenLevels(n)
    for a in range(1, data.n_treatments + 1):
        learner = make_learner()
        learners.append(learner)
        rows = data.pair(a)
        label = data.indicator(a)
        groups = cohort.arm_groups(a)
        if data.pattern_mode:
            assert data.miss is not None
            q, n_patterns = _pattern_propensities(
                data.X,
                label,
                rows,
                data.miss,
                learner,
                data.layout,
                unseen=unseen,
                metric_name=metric_name,
                treatment_groups=groups,
                control_group=control_group,
                method="IPTW",
            )
        else:
            if data.allow_mode:
                assert data.nan_cols is not None
                _probe_nan_capability(
                    unwrap(learner),
                    data.layout.encoded_width(),
                    data.layout.expand_columns(data.nan_cols),
                    role="propensity",
                    metric_name=metric_name,
                    treatment_groups=groups,
                    control_group=control_group,
                    method="IPTW",
                )
            q = _fit_predict(
                learner,
                data.X[rows],
                label[rows],
                data.X,
                cohort=cohort,
                a=a,
                stage=f"the in-sample propensity fit{_comparison_suffix(cohort, a)}",
                what="propensity",
                allow_mode=data.allow_mode,
            )
            unseen.record(learner, data.X)
        _finite_guard(
            q,
            what="propensity",
            learner_name=learner_name(learner),
            metric_name=metric_name,
            treatment_groups=groups,
            control_group=control_group,
            method="IPTW",
            allow_mode=data.allow_mode,
        )
        _propensity_range_guard(
            q,
            learner_name=learner_name(learner),
            metric_name=metric_name,
            treatment_groups=groups,
            control_group=control_group,
            method="IPTW",
        )
        conditional[:, a - 1] = q
    if data.pattern_mode:
        assert data.miss is not None
        data.X, data.layout, _ = _impute_with_indicators(
            data.X,
            data.layout,
            data.miss,
            metric_name=metric_name,
            treatment_groups=cohort.treatment_groups,
            control_group=control_group,
            method="IPTW",
        )
        data.note = (
            f"propensity fit separately per missingness pattern "
            f"({n_patterns} patterns; generalized propensity e(X_observed, pattern))"
        )
    _disclose_unseen_levels(cohort, data, unseen)
    return NuisancePredictions(conditional, _couple_propensities(conditional), {}), learners


def _iptw_contrast(
    request: AdjustedContrastRequest,
    *,
    data: AdjustmentData,
    support: ClusterSupport,
    mu1: float,
    mu0: float,
    psi1: np.ndarray,
    psi0: np.ndarray,
    note: str,
) -> LiftEstimate:
    """Infer one treatment's contrast from its arm and the shared control's
    Hájek means and aligned cohort influences, reduced on `support`: the rows
    those influences can be nonzero on and their clusters."""
    tau = mu1 - mu0
    if_tau = psi1 - psi0
    abs_scores = _score_stats(
        if_tau,
        metric=request.metric.name,
        contrast=request.treatment_group,
        support=support,
    )
    resolved = resolve_value_scale(
        value_scale=request.value_scale,
        prior=request.prior,
        raw_point=float(tau),
        raw_scores=abs_scores,
        mu0=float(mu0),
        mu0_se=lambda: _score_stats(
            psi0,
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=support,
        ).se(),
        lift_fn=lambda mu0_value: tau / mu0_value,
        lift_scores_fn=lambda mu0_value: _score_stats(
            ((psi1 - psi0) - (tau / mu0_value) * psi0) / mu0_value,
            metric=request.metric.name,
            contrast=request.treatment_group,
            support=support,
        ),
        refuse_near_zero=lambda mu0_value, se: _refuse_near_zero_adjustment_denominator(
            request, mu0_value, se, label="IPTW control mean"
        ),
    )
    point, scores = resolved.point, resolved.scores
    abs_diff, abs_se = resolved.abs_diff, resolved.abs_se
    if request.value_scale != "absolute":
        joint_result = _joint_reference_from_influences(
            if_tau,
            psi0,
            numerator_point=float(tau),
            denominator_point=float(mu0),
            supports=(support, support),
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
        note=note,
        n_clusters=support.k,
        dof=None,
        posterior=(resolved.posterior_point, resolved.posterior_scores),
        joint_result=joint_result,
    )


def _iptw_cohort(cohort: AdjustmentCohort, factory: Callable[[], Learner]) -> list[LiftEstimate]:
    """Estimate every treatment's IPTW contrast over one shared cohort.

    Fixed-propensity influences vanish outside their own arm, so each
    contrast reduces on its comparison's rows and clusters; the fitted
    logistic correction carries every comparison model's estimating
    equations onto every cohort row, so those contrasts reduce over the
    cohort's clusters."""
    data = _prepare_adjustment_cohort(cohort)
    data = _resolve_adjustment_missingness(cohort, data)
    nuisance, learners = _fit_propensities(cohort, data, factory)
    data, nuisance = _apply_overlap_gate(cohort, data, nuisance)
    p = nuisance.marginal
    n = data.arm.shape[0]
    arms = range(data.n_treatments + 1)
    indicators = [data.indicator(t) for t in arms]
    weights = [indicators[t] / p[:, t] for t in arms]
    clusters = _cohort_clusters(data)
    supports = [
        _validate_contrast(cohort, data, a, weights[a] + weights[0], clusters)
        for a in range(1, data.n_treatments + 1)
    ]
    big_w = [w.sum() for w in weights]
    mu = [(weights[t] * data.y).sum() / big_w[t] for t in arms]
    fixed = np.empty((n, len(weights)), order="F")
    for t in arms:
        fixed[:, t] = (n / big_w[t]) * indicators[t] * (data.y - mu[t]) / p[:, t]
    native = (
        not data.pattern_mode
        and not data.overlap_trimmed
        and all(type(unwrap(learner)) is LogisticPropensity for learner in learners)
    )
    if native:
        influences = _coupled_hajek_mean_influences(
            cast(Sequence[LogisticPropensity], [unwrap(learner) for learner in learners]),
            [fitted_design(learner, data.X) for learner in learners],
            data.arm,
            nuisance.conditional,
            p,
            fixed,
        )
        nuisance_note = _FITTED_LOGISTIC_NOTE
        supports = [clusters] * data.n_treatments
    else:
        influences = fixed
        nuisance_note = _FIXED_PROPENSITY_NOTE
    note = f"{data.note}; {nuisance_note}" if data.note else nuisance_note
    summary = [
        _weight_summary(w, clusters.inv, clusters.k)
        if clusters.inv is not None
        else _weight_summary(w)
        for w in weights
    ]
    definition = "unclipped inverse marginal propensity I(A=a)/p_a(X), retained cohort"
    grain = "cluster" if clusters.inv is not None else "unit"
    return [
        _iptw_contrast(
            request,
            data=data,
            support=support,
            mu1=float(mu[a]),
            mu0=float(mu[0]),
            psi1=influences[:, a],
            psi0=influences[:, 0],
            note=note,
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
        for a, (request, support) in enumerate(zip(cohort.requests, supports, strict=True), start=1)
    ]


def iptw_estimate(
    src: MomentSource,
    metric: Metric,
    design: Observational,
    *,
    learner: Callable[[], Learner] | None = None,
    prior: Prior | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    null_lift: float = 0.0,
    null_abs: float | None = None,
    value_scale: ValueScale = "relative",
    preferred_direction: PreferredDirection | None = None,
    moment_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[LiftEstimate]:
    """IPTW (Hajek self-normalized) estimate of relative lift, one entry per
    non-control treatment arm present for *metric*. See the module
    docstring for the Hajek-vs-Horvitz-Thompson choice. Registered under
    `Method.name == "iptw"`, `estimate_ate`'s default.

    Every arm's mean targets the whole eligible cohort: one learner per
    treatment is fitted on that treatment and control, its conditional
    propensity is predicted for every cohort unit, and the conditional odds
    are coupled into marginal arm propensities. The default untrimmed
    logistic fit carries its estimating equations into the covariance across
    all of those models.

    A cluster declared on the source (`src.cluster`, e.g.
    `from_unit_summary(cluster=...)`) switches the SE to the Liang-Zeger
    reduction over per-cluster influence totals with an asymptotic Normal
    reference. Each comparison needs at least two distinct clusters among its
    own treatment and control units, and two per arm when those clusters are
    arm-pure; below forty such clusters an advisory warning is emitted. A
    fixed-propensity contrast (generic learner, pattern or trimmed fit) is
    reduced over those comparison clusters; the fitted logistic correction
    reaches every cohort row, so its contrasts reduce over every cohort
    cluster.
    Read the cluster identity from the source, never a kwarg: a clustered
    source can never be served an iid SE by a direct call."""
    requests, learners = _prepare_adjustment_requests(
        src,
        metric,
        design,
        method="IPTW",
        covariates=list(design.adjustment.covariates),
        learner_roles=("learner",),
        method_fields={"learner": "propensity_learner"},
        learner=learner,
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
    active_learner = learners["learner"] or LogisticPropensity
    return _iptw_cohort(AdjustmentCohort(tuple(requests)), active_learner)


ADJUSTMENTS.register("iptw", iptw_estimate)
