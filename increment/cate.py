"""Public adapters: fit, validate, and deploy a CATE model from a unit-grain source.

Pulls one row per unit from MomentSource.unit_frame and hands the
columns to increment.estimation.cate and increment.estimation.targeting
as plain numpy arrays; lives outside increment/estimation/ because it
touches narwhals frames, keeping the estimation layer pure math.

estimate_cate fits; validate_cate is the gate saying whether the fit's
heterogeneity survives units it never saw. targeting_rule is the
decision that gate serves: treat a pre-committed top fraction, or fall
back to the simple policy.

``estimate_cate`` remains randomized-only. Validation and targeting also
support observational designs by cross-fitting a doubly robust score over
the declared adjustment set and enforcing its overlap gate.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import narwhals as nw
import numpy as np

from increment._identity import canonical_id_strings
from increment.errors import (
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._adjust.learners import LogisticPropensity, RidgeOutcome
from increment.estimation._deployment import resolve_deploy_grain
from increment.estimation.cate import (
    CateResult,
    Covariate,
    _validate_cluster_weight,
    fit_cate,
)
from increment.estimation.targeting import (
    CateValidation,
    ClusterBootstrap,
    PsiFn,
    TargetingRule,
    TargetingSelection,
    _dr_psi,
    _ipw_psi_fn,
    _prepare_adjustment_columns,
    select_targeting_rule_arrays,
    targeting_rule_arrays,
    validate_cate_arrays,
)
from increment.sources import MomentSource

if TYPE_CHECKING:
    from increment.semantics.design import Design, Observational
    from increment.semantics.models import Metric


_RANDOMIZED_ONLY = RefusalSpec(
    "cate.identification.randomized_only",
    InvalidRequestError,
    template="{caller} requires a randomized MomentSource (source.context.design.mechanism == 'randomized'), got {mechanism!r} -- estimate_cate's per-unit predictions do not have a doubly robust observational implementation",
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "cate.identification.unsupported_mechanism": "{caller} does not support identification mechanism {mechanism!r}; supported mechanisms are {supported}",
        "cate.identification.unsupported_missing_policy": "{caller} does not implement observational adjustment missing={missing!r}; this CATE path currently requires missing='refuse'",
        "cate.identification.unsupported_max_smd": "{caller} does not implement the observational max_smd={max_smd!r} balance gate",
        "cate.metric_declared_source": "metric {metric!r} is not declared on this source; available: {available}",
        "cate.does_support_ratio": RefusalSpec(
            "cate.does_support_ratio",
            UnsupportedRequestError,
            template="{caller} does not support ratio metric {metric!r}: conditional ratio effects require modeling denominator variation. Use readouts.run() for the arm-level ratio effect, or model numerator and denominator separately.",
        ),
        "cate.cate_supported_quantile": RefusalSpec(
            "cate.cate_supported_quantile",
            UnsupportedRequestError,
            template="CATE is not supported for quantile metric {metric!r} -- the per-unit outcome is the unit's total, so the fitted effect would be a conditional MEAN effect reported under a quantile metric's name. Declare the column as a mean metric instead",
        ),
        "cate.control_arm_present": "control arm {control!r} is not present in {metric!r}; arms are {arms}",
        "cate.contrasts_exactly_two": "{caller} contrasts exactly two arms, but {metric!r} has {n_arms}: {arms}. Against control {control!r} the treatment indicator would pool {others} into a single arm, which estimates a different quantity. Restrict the source to two arms and call once per contrast",
        "cate.cluster.covariate_conflict": "{caller}: {column!r} is this source's declared cluster identity (source.context.cluster) and cannot also be requested as an interact/adjust covariate or observational adjustment column, even though it may look numeric -- cluster identity is metadata, never a model feature.",
        "cate.cluster.randomization_arm_purity": "{caller}: cluster {cluster!r} contains both treatment arms ({values}) under a declared cluster-randomized source (source.context.design.mechanism == 'randomized' with a declared cluster) -- randomization at the cluster grain requires every cluster to be a pure arm. An observational design (declaring the dependence honestly) preserves mixed-treatment clusters instead.",
        "cate.cluster.identity_unavailable": "{caller}: source.context.cluster == {column!r} declares a cluster identity, but unit_frame served no aligned cluster_id column -- a nonconformant MomentSource implementation. Refusing rather than silently treating this population as unclustered.",
        "cate.cluster.intervention_grain_unavailable": "{caller}: source.context.intervention_grain == 'cluster' declares a whole-cluster policy deployment, but this source resolved no cluster identity (source.context.cluster is None) -- there is no cluster grain to intervene on.",
    },
)
_raise = raiser(_REFUSALS)


def _coerce(covariates: Sequence[Covariate | str]) -> tuple[Covariate, ...]:
    """A bare string names a continuous covariate; a typed spec passes through."""
    return tuple(Covariate(name=c) if isinstance(c, str) else c for c in covariates)


def _requested_columns(*groups: Sequence[Covariate]) -> list[str]:
    """Every covariate column to pull, deduplicated, first-occurrence order.

    ``unit_frame`` selects the names verbatim, so a column named by both
    *interact* and *adjust* must be asked for once.
    """
    seen: dict[str, None] = {}
    for group in groups:
        for cov in group:
            seen.setdefault(cov.name, None)
    return list(seen)


def _resolve_metric(source: MomentSource, metric: str) -> Metric:
    """The declared metric named *metric*, or a ValueError naming it."""
    declared = source.context.metrics
    for candidate in declared:
        if candidate.name == metric:
            return candidate
    available = sorted(m.name for m in declared)
    _raise("cate.metric_declared_source", metric=metric, available=available)


class _UnitDesign(NamedTuple):
    """Unit-grain arrays and the frame they came from."""

    frame: nw.DataFrame
    y: np.ndarray
    d: np.ndarray
    cols: dict[str, np.ndarray]
    identification: Design | None
    #: Canonicalized, aligned cluster identity -- `None` when the source
    #: declares no `cluster`. Metadata only: never entered as a model column.
    cluster_ids: np.ndarray | None
    #: The declared whole-cluster policy-deployment grain
    #: (`source.context.intervention_grain`), transported independently of
    #: `cluster_ids` and of any covariance clustering.
    intervention_grain: Literal["unit", "cluster"]


def _refuse_impure_clusters(cluster_ids: np.ndarray, d: np.ndarray, *, caller: str) -> None:
    """Refuse a cluster spanning both arms under declared cluster-randomization.

    Mirrors `increment.estimation.crossfit.check_cluster_atomic`'s sorted-
    boundary scan, applied to arm purity rather than a train/test split.
    """
    order = np.argsort(cluster_ids)
    sorted_ids = cluster_ids[order]
    sorted_d = d[order]
    boundary = np.empty(sorted_ids.shape, dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_ids[1:] != sorted_ids[:-1]
    changed = np.zeros(sorted_ids.shape, dtype=bool)
    changed[1:] = sorted_d[1:] != sorted_d[:-1]
    impure = changed & ~boundary
    if impure.any():
        i = int(np.flatnonzero(impure)[0])
        _raise(
            "cate.cluster.randomization_arm_purity",
            caller=caller,
            cluster=sorted_ids[i],
            values=(sorted_d[i - 1].item(), sorted_d[i].item()),
        )


def _unit_design(
    source: MomentSource,
    metric: str,
    *,
    control: str,
    covariates: Sequence[str],
    caller: str,
    permit_observational: bool,
) -> _UnitDesign:
    """Outcome, treatment indicator and covariate columns for *metric*.

    Every public entry point reads the same rows under the same contract;
    *caller* names the refusing function in error messages.
    """
    design = source.context.design
    mechanism = design.mechanism if design is not None else None
    if permit_observational:
        if mechanism not in ("randomized", "observational"):
            _raise(
                "cate.identification.unsupported_mechanism",
                caller=caller,
                mechanism=mechanism,
                supported=("randomized", "observational"),
            )
        if mechanism == "observational":
            assert design is not None
            observational = cast("Observational", design)
            if observational.adjustment.missing != "refuse":
                _raise(
                    "cate.identification.unsupported_missing_policy",
                    caller=caller,
                    missing=observational.adjustment.missing,
                )
            if observational.gate.max_smd is not None:
                _raise(
                    "cate.identification.unsupported_max_smd",
                    caller=caller,
                    max_smd=observational.gate.max_smd,
                )
    elif mechanism != "randomized":
        refuse(_RANDOMIZED_ONLY, caller=caller, mechanism=mechanism)
    requested = list(covariates)
    if design is not None and design.mechanism == "observational":
        requested = list(dict.fromkeys((*requested, *design.adjustment.covariates)))
    cluster_column = source.context.cluster
    if cluster_column is not None and cluster_column in requested:
        _raise("cate.cluster.covariate_conflict", caller=caller, column=cluster_column)
    resolved = _resolve_metric(source, metric)
    if resolved.type == "ratio":
        _raise("cate.does_support_ratio", caller=caller, metric=metric)
    if resolved.type == "quantile":
        _raise("cate.cate_supported_quantile", metric=metric)
    frame = nw.from_native(source.unit_frame(resolved, covariates=requested), eager_only=True)

    if design is not None and design.mechanism == "observational":
        numeric_adjustments = [
            name for name in design.adjustment.covariates if frame.schema[name].is_numeric()
        ]
        frame = frame.with_columns(*(nw.col(name).cast(nw.Float64) for name in numeric_adjustments))
    # A dictionary-encoded column's `to_numpy()` can rewrite a null level as
    # a neighbour's; `to_list()` keeps every null null on every backend.
    textual = {
        name for name in requested if frame.schema[name] in (nw.String, nw.Categorical, nw.Enum)
    }

    group_values = frame["group_id"].to_list()
    arms = sorted(set(group_values), key=lambda v: (v is None, str(v)))
    if control not in arms:
        _raise("cate.control_arm_present", control=control, metric=metric, arms=arms)
    if len(arms) != 2:
        others = [a for a in arms if a != control]
        _raise(
            "cate.contrasts_exactly_two",
            caller=caller,
            metric=metric,
            n_arms=len(arms),
            arms=arms,
            control=control,
            others=others,
        )
    # .to_numpy() avoids a Python-object round trip per column at
    # warehouse scale; nulls surface as NaN, caught by the null check below.
    y = np.asarray(frame["y"].to_numpy(), dtype=float)
    d = np.fromiter((g != control for g in group_values), dtype=float, count=len(group_values))
    cols = {
        name: (
            np.asarray(frame[name].to_list(), dtype=object)
            if name in textual
            else np.asarray(frame[name].to_numpy())
        )
        for name in requested
    }
    if design is not None and design.mechanism == "observational":
        cols = _prepare_adjustment_columns(cols, design.adjustment.covariates, caller=caller)

    cluster_ids: np.ndarray | None = None
    if cluster_column is not None:
        # A declared cluster is a hard requirement on every row's identity:
        # silently treating a missing `cluster_id` column as "unclustered"
        # would make cluster-atomic splitting/fold assignment and honest
        # validation silently unit-independent for data that is not.
        if "cluster_id" not in frame.columns:
            _raise("cate.cluster.identity_unavailable", caller=caller, column=cluster_column)
        cluster_ids = canonical_id_strings(
            np.asarray(frame["cluster_id"].to_numpy()), what="cluster_ids"
        )
        # Declared cluster-randomization (a randomized design over a
        # coarser cluster grain) requires every cluster to be a pure arm;
        # a declared observational design preserves mixed-treatment
        # dependence clusters instead (see the module docstring).
        if mechanism == "randomized":
            _refuse_impure_clusters(cluster_ids, d, caller=caller)

    intervention_grain = source.context.intervention_grain
    if intervention_grain == "cluster" and cluster_ids is None:
        _raise("cate.cluster.intervention_grain_unavailable", caller=caller)

    return _UnitDesign(
        frame=frame,
        y=y,
        d=d,
        cols=cols,
        identification=design,
        cluster_ids=cluster_ids,
        intervention_grain=intervention_grain,
    )


_DR_CROSSFIT_FOLDS = 5
_DR_CROSSFIT_SEED = 0


def _score_plan(
    identification: Design | None,
) -> tuple[tuple[str, ...], PsiFn, Literal["welch", "score"]]:
    """Score construction and arm summary implied by identification.

    Clustered array adapters freeze this built-in DR partial using training-only
    fits before evaluating holdout scores; unclustered adapters retain crossfit.
    """
    if identification is not None and identification.mechanism == "observational":
        return (
            identification.adjustment.covariates,
            functools.partial(
                _dr_psi,
                propensity_learner=LogisticPropensity,
                outcome_learner=RidgeOutcome,
                folds=_DR_CROSSFIT_FOLDS,
                seed=_DR_CROSSFIT_SEED,
                gate=identification.gate,
            ),
            "score",
        )
    return (), _ipw_psi_fn, "welch"


def estimate_cate(
    source: MomentSource,
    metric: str,
    *,
    control: str,
    interact: Sequence[Covariate | str],
    adjust: Sequence[Covariate | str] = (),
    alpha: float = 0.05,
    ard: bool = False,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
) -> CateResult:
    """Estimate conditional average treatment effects for *metric*.

    Nothing this returns is validated: scoring units with it and
    reporting the top group's effect is exactly the in-sample
    fabrication ``validate_cate`` exists to catch - on data with no true
    heterogeneity, that top quintile reads 2.9x the true effect. Run
    ``validate_cate`` before any subgroup number leaves this function.

    ``source`` must serve unit-grain rows (``CapabilityError``
    otherwise). ``control`` names the control ``group_id``; the frame
    must contain exactly that arm plus one other. ``interact``
    covariates model effect heterogeneity; ``adjust`` covariates enter
    as main effects only, for precision without a heterogeneity claim.
    ``ard`` shrinks the interaction coefficients by evidence
    maximization, moving only the scored points - not
    ``ate``/``se``/``heterogeneity``.
    Returns effects on the metric's own absolute scale. Declared clusters use
    a weighted cluster-score sandwich with t(K-1) intervals; otherwise HC2.
    ``cluster_weight="member_count"`` weights units equally (the default);
    ``"equal"`` weights clusters equally and requires a declared cluster.
    Intervention grain never selects weighting. Cluster rank deficiency,
    single-cluster-only directions and unavailable uncertainty refuse with
    ``InvalidRequestError``; inference metadata live on the returned fit.
    Raises ``InvalidRequestError`` (``cate.identification.randomized_only``)
    for a source whose design is not randomized; ``ValueError`` for an
    undeclared metric, absent control, or a non-two-arm frame;
    ``NotImplementedError`` for ratio and quantile metrics.

    Covariates must be strictly pre-exposure; results do not compose
    with the package's default relative lifts (see
    ``increment.estimation.cate``).
    """
    _validate_cluster_weight(cluster_weight, clustered=source.context.cluster is not None)
    interact_covs = _coerce(interact)
    adjust_covs = _coerce(adjust)
    covariates = _requested_columns(interact_covs, adjust_covs)
    unit = _unit_design(
        source,
        metric,
        control=control,
        covariates=covariates,
        caller="estimate_cate",
        permit_observational=False,
    )
    return fit_cate(
        unit.y,
        unit.d,
        unit.cols,
        interact=interact_covs,
        adjust=adjust_covs,
        alpha=alpha,
        ard=ard,
        cluster_weight=cluster_weight,
        cluster_ids=unit.cluster_ids,
        intervention_grain=unit.intervention_grain,
    )


def validate_cate(
    source: MomentSource,
    metric: str,
    *,
    control: str,
    interact: Sequence[Covariate | str],
    adjust: Sequence[Covariate | str] = (),
    n_groups: int = 5,
    alpha: float = 0.05,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap: ClusterBootstrap | None = None,
    include_evaluation_population: bool = False,
) -> CateValidation:
    """Fit CATE on one honest partition and validate it on the other.

    The gate for every heterogeneity claim: a CATE fit always hands back
    a winner, so the top group's in-sample effect is not evidence of
    anything. Splits the units by a content hash of their id, fits on
    one half (whole clusters when declared), and reports sorted-group effects,
    rank tests, and a CLAN profile computed entirely on the other.
    ``CateValidation.passed`` is the verdict; while false, the only defensible
    number is the average effect.

    Randomized sources use raw arm contrasts. Observational sources use
    cross-fitted doubly robust scores over the design's declared adjustment
    set; a numeric adjustment column enters as it is and a string column as
    modal-reference level indicators fitted inside every nuisance fit (a
    null level refuses, as this path supports ``missing="refuse"`` only).
    Its overlap gate can refuse or trim the
    reported population. ``n_groups`` sets how many predicted-effect groups
    to cut the holdout into. ``alpha`` is two-sided for every reported
    interval; the rank tests are one-sided against it, and ``passed`` is
    ``autoc.p_value < alpha``.

    Returns holdout-only numbers on the metric's own absolute scale.
    Units are keyed by ``unit_frame``'s ``unit_id`` (already cast to
    String), so the same logical unit lands in the same half across
    runs. There is deliberately no ``ard=`` here: this gate's null size
    was measured on the unclustered unpenalized score (4.8% against a nominal 5%),
    and shrinkage is a reporting choice for ``estimate_cate``, not a
    knob on the test.

    Missing or unsupported identification raises
    ``cate.identification.unsupported_mechanism``. Observational policies
    other than ``missing="refuse"`` or with a non-null ``max_smd`` raise
    ``cate.identification.unsupported_missing_policy`` or
    ``cate.identification.unsupported_max_smd`` before reading data.
    Declared clusters use ``cluster_weight="member_count"`` (unit weights)
    or ``"equal"`` (inverse cluster-size weights). Clustered rank, GATES and
    CLAN intervals contain both a heldout-only whole-cluster bootstrap-t and a
    delete-one-cluster jackknife-t interval, conditional on training-frozen
    nuisances; ``bootstrap`` (default
    ``ClusterBootstrap(seed=0, repetitions=999)``) is recorded on results.
    Missing uncertainty has a nullable numeric field and an
    ``unavailable_reason`` code.
    ``n_train`` and ``n_holdout`` count members; ``n_clusters`` counts the
    independent heldout clusters. The unclustered calibration above does not
    establish calibration of the cluster bootstrap at small cluster counts.
    """
    _validate_cluster_weight(cluster_weight, clustered=source.context.cluster is not None)
    bootstrap = (
        ClusterBootstrap() if bootstrap is None else ClusterBootstrap.model_validate(bootstrap)
    )
    interact_covs = _coerce(interact)
    adjust_covs = _coerce(adjust)
    covariates = _requested_columns(interact_covs, adjust_covs)
    unit = _unit_design(
        source,
        metric,
        control=control,
        covariates=covariates,
        caller="validate_cate",
        permit_observational=True,
    )
    adjustment, psi_fn, arm_summary = _score_plan(unit.identification)
    return validate_cate_arrays(
        unit.y,
        unit.d,
        unit.cols,
        np.asarray(unit.frame["unit_id"].to_numpy()),
        cluster_ids=unit.cluster_ids,
        cluster_weight=cluster_weight,
        bootstrap_seed=bootstrap.seed,
        bootstrap_repetitions=bootstrap.repetitions,
        interact=interact_covs,
        adjust=adjust_covs,
        n_groups=n_groups,
        alpha=alpha,
        adjustment=adjustment,
        psi_fn=psi_fn,
        arm_summary=arm_summary,
        intervention_grain=unit.intervention_grain,
        include_evaluation_population=include_evaluation_population,
    )


def targeting_rule(
    source: MomentSource,
    metric: str,
    *,
    control: str,
    interact: Sequence[Covariate | str],
    adjust: Sequence[Covariate | str] = (),
    fraction: float,
    alpha: float = 0.05,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap: ClusterBootstrap | None = None,
    deploy_grain: Literal["unit", "cluster"] | None = None,
    include_evaluation_population: bool = False,
) -> TargetingRule:
    """Decide whether to target the top *fraction* of units on *metric*.

    The question a heterogeneity analysis is actually asked is not
    "which group responded best" but "what rule should I deploy". Runs
    ``validate_cate``'s honest split; unless the gate passes, returns
    ``recommendation="simple"`` (treat everyone alike on the average
    effect). Only a passing gate gets a threshold, a policy value, and
    an uplift over the average.

    ``fraction`` is required with no default: pre-committing to the cut
    is the entire value of this function - choosing it after seeing
    ``validate_cate``'s group table biases the reported policy value
    upward by 60-120%. Identification and policy refusals follow
    ``validate_cate``, including observational adjustment and overlap rules.
    ``alpha`` gates the same ``autoc.p_value < alpha`` test. ``ard=`` is absent for the
    same reason it is absent from ``validate_cate``: the gate was
    calibrated on the unpenalized score.

    A failed gate is a result, not an exception: ``recommendation`` and
    the attached ``validation`` say why, and the three policy fields are
    ``None`` together so no caller reads an unsupported number.
    Conditional on a gate that passed by chance, ``policy_value`` runs
    high - unconditionally it is unbiased; treat a barely-passing gate
    as weak evidence for the magnitude, not just the ranking.
    Declared clusters use ``cluster_weight="member_count"`` (unit weights)
    or ``"equal"`` (inverse cluster-size weights). Clustered rank, GATES and
    CLAN intervals contain both a heldout-only whole-cluster bootstrap-t and a
    delete-one-cluster jackknife-t interval, conditional on training-frozen
    nuisances; ``bootstrap`` (default
    ``ClusterBootstrap(seed=0, repetitions=999)``) is recorded on results.
    Deployment defaults only from the declared intervention grain. Cluster
    policies pool member scores and take the longest feasible whole-cluster
    prefix, with canonical-ID tie breaks; ``cluster_weight`` determines both
    budget and evaluation mass. Fractions zero/one deploy nobody/everybody.
    The result stores requested and achieved shares and portable scoring state.
    ``predict`` applies the candidate policy; ``recommendation`` is its gate.
    Missing uncertainty has a nullable numeric field and an
    ``unavailable_reason`` code.
    """
    deploy_grain = resolve_deploy_grain(
        source.context.intervention_grain,
        deploy_grain,
        clustered=source.context.cluster is not None,
    )
    _validate_cluster_weight(cluster_weight, clustered=source.context.cluster is not None)
    bootstrap = (
        ClusterBootstrap() if bootstrap is None else ClusterBootstrap.model_validate(bootstrap)
    )
    interact_covs = _coerce(interact)
    adjust_covs = _coerce(adjust)
    covariates = _requested_columns(interact_covs, adjust_covs)
    unit = _unit_design(
        source,
        metric,
        control=control,
        covariates=covariates,
        caller="targeting_rule",
        permit_observational=True,
    )
    adjustment, psi_fn, arm_summary = _score_plan(unit.identification)
    return targeting_rule_arrays(
        unit.y,
        unit.d,
        unit.cols,
        np.asarray(unit.frame["unit_id"].to_numpy()),
        cluster_ids=unit.cluster_ids,
        cluster_weight=cluster_weight,
        bootstrap_seed=bootstrap.seed,
        bootstrap_repetitions=bootstrap.repetitions,
        interact=interact_covs,
        adjust=adjust_covs,
        fraction=fraction,
        alpha=alpha,
        adjustment=adjustment,
        psi_fn=psi_fn,
        arm_summary=arm_summary,
        intervention_grain=unit.intervention_grain,
        deploy_grain=deploy_grain,
        include_evaluation_population=include_evaluation_population,
    )


# Identification, selection and deployment controls remain explicit at the public boundary.
def select_targeting_rule(  # noqa: PLR0913
    source: MomentSource,
    metric: str,
    *,
    control: str,
    interact: Sequence[Covariate | str],
    adjust: Sequence[Covariate | str] = (),
    fractions: Sequence[float],
    cost_per_treated: float = 0.0,
    n_folds: int = 5,
    seed: int,
    alpha: float = 0.05,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap: ClusterBootstrap | None = None,
    deploy_grain: Literal["unit", "cluster"] | None = None,
    include_evaluation_population: bool = False,
) -> TargetingSelection:
    """Choose the share of units to target on *metric*, honestly.

    :func:`targeting_rule` demands a pre-committed fraction because choosing
    the cut after seeing results biases the reported policy value upward by
    60-120%.  This is the sanctioned way to CHOOSE that fraction: a seeded,
    arm-stratified half of the units is set aside untouched; on the other
    half, K-fold out-of-fold scores estimate each grid fraction's net
    benefit ``E[1{targeted} (tau - cost_per_treated)]``; the argmax is
    locked and then evaluated exactly once on the untouched half, through
    the same gate and policy numbers :func:`targeting_rule` reports.
    For observational sources, both the inner selection objective and the
    untouched outer report use the declared doubly robust score and overlap policy.

    *fractions*, in ``[0, 1]`` each, and *seed* are REQUIRED with no
    defaults: the grid and the split are pre-commitments.
    *cost_per_treated* is in the metric's own units per treated unit;
    at the default ``0.0`` the objective is total benefit, which favors
    wide fractions whenever the marginal unit's effect is positive.

    Raises exactly what :func:`targeting_rule` raises, plus ``ValueError``
    for an invalid grid or a fold too small to hold 2 units per arm.
    Declared clusters use ``cluster_weight="member_count"`` (unit weights)
    or ``"equal"`` (inverse cluster-size weights). Clustered rank, GATES and
    CLAN intervals contain both a heldout-only whole-cluster bootstrap-t and a
    delete-one-cluster jackknife-t interval, conditional on training-frozen
    nuisances; ``bootstrap`` (default
    ``ClusterBootstrap(seed=0, repetitions=999)``) is recorded on results.
    Deployment defaults only from the declared intervention grain. Cluster
    policies pool member scores and take the longest feasible whole-cluster
    prefix, with canonical-ID tie breaks; ``cluster_weight`` determines both
    budget and evaluation mass. Fractions zero/one deploy nobody/everybody.
    The result stores requested and achieved shares and portable scoring state.
    ``predict`` applies the candidate policy; ``recommendation`` is its gate.
    Missing uncertainty has a nullable numeric field and an
    ``unavailable_reason`` code.
    """
    deploy_grain = resolve_deploy_grain(
        source.context.intervention_grain,
        deploy_grain,
        clustered=source.context.cluster is not None,
    )
    _validate_cluster_weight(cluster_weight, clustered=source.context.cluster is not None)
    bootstrap = (
        ClusterBootstrap() if bootstrap is None else ClusterBootstrap.model_validate(bootstrap)
    )
    interact_covs = _coerce(interact)
    adjust_covs = _coerce(adjust)
    covariates = _requested_columns(interact_covs, adjust_covs)
    unit = _unit_design(
        source,
        metric,
        control=control,
        covariates=covariates,
        caller="select_targeting_rule",
        permit_observational=True,
    )
    adjustment, psi_fn, arm_summary = _score_plan(unit.identification)
    return select_targeting_rule_arrays(
        unit.y,
        unit.d,
        unit.cols,
        np.asarray(unit.frame["unit_id"].to_numpy()),
        cluster_ids=unit.cluster_ids,
        cluster_weight=cluster_weight,
        bootstrap_seed=bootstrap.seed,
        bootstrap_repetitions=bootstrap.repetitions,
        interact=interact_covs,
        adjust=adjust_covs,
        fractions=fractions,
        cost_per_treated=cost_per_treated,
        n_folds=n_folds,
        seed=seed,
        alpha=alpha,
        adjustment=adjustment,
        psi_fn=psi_fn,
        arm_summary=arm_summary,
        intervention_grain=unit.intervention_grain,
        deploy_grain=deploy_grain,
        include_evaluation_population=include_evaluation_population,
    )


__all__ = [
    "estimate_cate",
    "select_targeting_rule",
    "targeting_rule",
    "validate_cate",
]
