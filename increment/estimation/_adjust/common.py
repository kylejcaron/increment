"""Shared adjustment stages for observational contrast estimators.

Every requested treatment of one metric is compared with the declared control
over one eligible cohort: missingness, cross-fitting folds, nuisance fits,
overlap and cluster reductions are resolved once for that cohort. Treatment
versus control rows train each comparison's own propensity model; their
conditional propensities are coupled into marginal arm propensities that are
evaluated on every cohort row.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals as nw
import numpy as np

from increment._literals import PreferredDirection, ValueScale
from increment.errors import (
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation._adjust.encoding import (
    CovariateLayout,
    UnseenLevels,
    classify_objects,
    encoded_factory,
    fixed_design,
    is_null,
    learner_name,
    level_codes,
    unseen_levels_text,
)
from increment.estimation._adjust.overlap import (
    IdentificationError,
    SmdArms,
    _allow_fit_predict,
    _allow_interaction_smds,
    _allow_nan_mask,
    _allow_note,
    _allow_requires_learner_refusal,
    _cluster_index,
    _clustered_fields,
    _contrast_text,
    _dyadic_integers,
    _finite_guard,
    _fold_ids,
    _impute_with_indicators,
    _infeasible_band_note,
    _missing_covariate_refusal,
    _moment_dicts,
    _overlap_refusal_message,
    _probe_nan_capability,
    _propensity_range_guard,
    _propensity_ranges,
    _restrict_cluster_index,
    _validate_pair_cluster_support,
    _weighted_smd,
)
from increment.estimation._adjust.overlap import (
    _raise as _overlap_raise,
)
from increment.estimation._adjust.value_scale import _ABSOLUTE_REMEDY, _compose_absolute_note
from increment.estimation.armstats import ScoreStats
from increment.estimation.engine import check_total_clusters
from increment.estimation.inference import infer_ate

if TYPE_CHECKING:
    from increment.estimation._adjust.learners import Learner
    from increment.estimation.armstats import ScoreStats
    from increment.estimation.inference import Prior
    from increment.estimation.results import (
        JointContrastReference,
        LiftEstimate,
        RelativeUnavailableReason,
    )
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Callable[..., str]
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    # +1 absorbs this helper's own frame; errors.warn() absorbs its own.
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "estimation.adjust_common.covariate_balance_advisory",
    IncrementWarning,
    lambda *, method, metric_name, treatment_group, control_group, named, detail: (
        f"{method} covariate balance advisory for metric {metric_name!r} "
        f"(contrast {treatment_group!r} vs {control_group!r}): {named} "
        f"have |SMD| > 0.1 {detail}"
    ),
)


_register_warning(
    "estimation.adjust_common.unseen_level_advisory",
    IncrementWarning,
    lambda *, method, metric_name, treatment_groups, control_group, unseen, n: (
        f"{method} unseen-level advisory for metric {metric_name!r} "
        f"({_contrast_text(treatment_groups, control_group)}): {unseen_levels_text(unseen, n)}"
    ),
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.adjust_common.supported_ratio_metric": RefusalSpec(
            "estimation.adjust_common.supported_ratio_metric",
            UnsupportedRequestError,
            template="{method} is not supported for ratio metric {metric!r} -- {method} reweights or residualizes a single per-unit outcome, and a ratio's two components would need the weights applied jointly with their covariance retained, which is not implemented for this method. Ratio metrics ARE supported under randomized CUPED (Method(name='cuped', variance_reduction='cuped') with a covariate= column), which adjusts the numerator and the denominator with a slope each and keeps their induced covariance. Otherwise adjust the numerator and denominator as separate mean metrics.",
            # Every raise site names the plan role and family; None where unknown.
            keys=frozenset({"role", "family", "correction"}),
        ),
        "estimation.adjust_common.control_group_found": "control_group {control_group!r} not found in {metric!r} groups. Available groups: {available}",
        "estimation.adjust_common.needs_least_folds": "{method} needs at least 2*folds={needed} units to cross-fit for metric {metric!r} ({contrast}); got {n}. Lower `folds` or supply more data.",
        "estimation.adjust_common.fold_would_train": "{method} fold {j} would train on zero units for metric {metric!r} ({contrast}) -- the unit-id hash collapsed every unit into one fold; lower `folds`.",
        "estimation.adjust_common.fold_training_split": "{method} fold {j} training split has only one arm for metric {metric!r} ({contrast}) -- cannot fit the propensity/outcome model(s); lower `folds` or supply more data.",
        "estimation.adjust_common.covariate_dtype": "{method}: adjustment covariate {covariate!r} has dtype {dtype} for metric {metric!r} ({contrast}) -- an adjustment covariate is numeric (int/float/bool) or categorical (string); a date, time, nested or mixed-type column carries no adjustment meaning. Derive a numeric or categorical pre-exposure column upstream.",
        "estimation.adjust_common.statistically_indistinguishable_from": RefusalSpec(
            "estimation.adjust_common.statistically_indistinguishable_from",
            InvalidRequestError,
            lambda *, label, denominator, se, metric, treatment, control, estimator: (
                f"{label} {denominator:.3g} is statistically indistinguishable from 0 "
                f"(SE {se:.3g}) for metric {metric!r} (contrast "
                f"{treatment!r} vs {control!r}) -- relative "
                f"lift {'theta' if estimator == 'DML' else 'tau'} / mu0 is not "
                "identified on this scale. " + _ABSOLUTE_REMEDY.format(name=metric)
            ),
        ),
    },
)
_raise = raiser(_REFUSALS)
SUPPORTED_RATIO_METRIC = _REFUSALS["estimation.adjust_common.supported_ratio_metric"]


def _prediction_shape_guard(
    pred: object,
    *,
    expected_n: int,
    learner_name: str,
    what: str,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> np.ndarray:
    """Coerce a learner's prediction to a finite-shaped array, or refuse.

    A scalar (Python float, 0-d array) or wrong-length array silently
    broadcasts against a later elementwise op instead of raising -- a
    learner returning a constant reads as perfect balance, not a broken fit.
    """
    arr = np.asarray(pred, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != expected_n:
        raise IdentificationError(
            f"{method}: learner {learner_name} returned a {what} prediction of "
            f"shape {arr.shape} for metric {metric_name!r} "
            f"({_contrast_text(treatment_groups, control_group)}), expected a 1-D array of "
            f"length {expected_n} -- a scalar or mis-shaped prediction silently "
            "broadcasts against later elementwise ops",
            code="estimation.adjust.learner.prediction_shape",
            context={
                "learner_name": learner_name,
                "what": what,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "method": method,
                "shape": arr.shape,
                "expected_n": expected_n,
            },
        )
    return arr


@dataclass(frozen=True, slots=True)
class AdjustedContrastRequest:
    """Immutable context shared by one adjusted treatment contrast."""

    frame: nw.DataFrame
    metric: Metric
    design: Observational
    control_group: str
    treatment_group: str
    control_native: object
    treatment_native: object
    covariates: list[str]
    prior: Prior | None
    alpha: float
    alternative: str
    null_lift: float
    null_abs: float | None
    value_scale: ValueScale
    preferred_direction: PreferredDirection | None
    cluster: str | None
    method: str


@dataclass(frozen=True, slots=True)
class AdjustmentCohort:
    """Every requested treatment comparison of one metric, sharing one cohort.

    Arm code 0 is the declared control and code ``a`` is ``requests[a - 1]``'s
    treatment. Each comparison's contrast is evaluated over the same eligible
    cohort (the source's complete unit frame for the metric), not over its own
    treatment/control rows.
    """

    requests: tuple[AdjustedContrastRequest, ...]

    @property
    def lead(self) -> AdjustedContrastRequest:
        return self.requests[0]

    @property
    def metric(self) -> Metric:
        return self.lead.metric

    @property
    def design(self) -> Observational:
        return self.lead.design

    @property
    def method(self) -> str:
        return self.lead.method

    @property
    def control_group(self) -> str:
        return self.lead.control_group

    @property
    def treatment_groups(self) -> tuple[str, ...]:
        return tuple(request.treatment_group for request in self.requests)

    @property
    def groups(self) -> tuple[str, ...]:
        """Group label by arm code, control first."""
        return (self.control_group, *self.treatment_groups)

    def arm_groups(self, a: int) -> tuple[str, ...]:
        """Treatment groups a nuisance fit for arm code ``a`` serves: its own
        comparison, or every comparison for the shared control arm."""
        return self.treatment_groups if a == 0 else (self.requests[a - 1].treatment_group,)


@dataclass(frozen=True, slots=True)
class OutcomeRole:
    """One cross-fitted outcome regression.

    ``kind="arm"`` trains on arm ``arm``'s training rows and predicts every
    heldout cohort row (AIPW's per-arm means; DML's control-arm mean).
    ``kind="pair"`` trains on the rows of treatment ``arm`` and control and
    predicts that comparison's heldout rows (DML's pooled partialling-out
    regression, whose score is zero outside the comparison).
    """

    kind: Literal["arm", "pair"]
    arm: int


@dataclass(frozen=True, slots=True)
class NuisanceFoldSpec:
    """Learner factories and outcome roles for `_crossfit_nuisances`.

    One propensity learner per treatment is fitted on that treatment and
    control. `outcome_roles` are instantiated and fitted per fold after the
    propensities, in order; `deferred_roles` are fitted only after every base
    fold, on the same saved folds, so an optional nuisance cannot perturb the
    base fit schedule of a factory with shared state.
    """

    propensity_factory: Callable[[], Learner]
    outcome_factory: Callable[[], Learner]
    outcome_roles: tuple[OutcomeRole, ...]
    deferred_roles: tuple[OutcomeRole, ...]


def _prepare_adjustment_requests(
    src: MomentSource,
    metric: Metric,
    design: Observational,
    *,
    method: str,
    covariates: list[str],
    learner_roles: tuple[str, ...],
    method_fields: Mapping[str, str] | None = None,
    **contrast_kwargs: Any,
) -> tuple[list[AdjustedContrastRequest], dict[str, Callable[[], Learner] | None]]:
    """Shared ratio/missing-allow/moments/frame preamble for `aipw_estimate`,
    `dml_estimate`, `iptw_estimate`. `contrast_kwargs` carries each
    `learner_roles` name (its factory or `None`) plus `AdjustedContrastRequest`'s
    remaining fields (`prior`, `alpha`, `alternative`, `null_lift`, `null_abs`,
    `value_scale`, `preferred_direction`); the learner entries are popped out
    and returned separately so the caller applies its own per-role default.
    `method_fields` maps each `learner_roles` name to the `Method(...)`
    constructor field a caller would actually set in the `missing='allow'`
    refusal -- identity for AIPW/DML (whose kwarg names already match
    `Method.propensity_learner`/`outcome_learner`), but
    `{"learner": "propensity_learner"}` for IPTW, whose own kwarg is named
    `learner` while `Method` has no such field."""
    if metric.type == "ratio":
        _raise(
            "estimation.adjust_common.supported_ratio_metric",
            method=method,
            metric=metric.name,
            role=None,
            family=None,
            correction=None,
        )
    learners = {name: contrast_kwargs.pop(name, None) for name in learner_roles}
    if design.adjustment.missing == "allow":
        fields = method_fields or {name: name for name in learner_roles}
        defaulted = [fields[name] for name in learner_roles if learners[name] is None]
        if defaulted:
            raise _allow_requires_learner_refusal(
                method=method, metric_name=metric.name, defaulted=defaulted
            )
    if src.context.cluster is not None and contrast_kwargs.get("prior") is not None:
        from increment.estimation.inference import _raise as inference_refuse

        inference_refuse("estimation.inference.prior_excludes_cluster_robust_t")
    control_group = design.control_group
    moment_rows = _moment_dicts(cast("list[Mapping[str, Any]]", src.moments(metric)))
    group_native: dict[str, object] = {}
    group_n: dict[str, int] = {}
    for row in moment_rows:
        key = str(row["group_id"])
        group_native[key] = row["group_id"]
        group_n[key] = int(row["n"])
    if control_group not in group_native:
        available = sorted(group_native)
        _raise(
            "estimation.adjust_common.control_group_found",
            available=available,
            control_group=control_group,
            metric=metric.name,
        )
    treatment_groups = sorted(g for g in group_native if g != control_group)
    cluster = src.context.cluster
    frame = nw.from_native(src.unit_frame(metric, covariates=covariates), eager_only=True)
    requests = [
        AdjustedContrastRequest(
            frame=frame,
            metric=metric,
            design=design,
            control_group=control_group,
            treatment_group=treatment_group,
            control_native=group_native[control_group],
            treatment_native=group_native[treatment_group],
            covariates=covariates,
            cluster=cluster,
            method=method,
            **contrast_kwargs,
        )
        for treatment_group in treatment_groups
        if group_n[treatment_group] != 0
    ]
    return requests, learners


@dataclass(slots=True)
class AdjustmentData:
    """Working cohort arrays and metadata after frame and missingness stages.

    `arm` holds arm codes (0 control, ``a`` the cohort's ``a``-th treatment).
    `X` is one float matrix: numeric covariates as they are, categorical
    covariates as level codes described by `layout`; every learner fits its
    own level encoding on its own training rows (`encoding.py`).
    `clusters` is the retained rows' cluster index once `_cohort_clusters`
    has built it; the overlap gate restricts it rather than sorting again.
    """

    arm: np.ndarray
    n_treatments: int
    y: np.ndarray
    X: np.ndarray
    unit_id: np.ndarray
    labels: np.ndarray | None
    layout: CovariateLayout
    population: str | None = None
    note: str | None = None
    allow_mode: bool = False
    X_gate: np.ndarray | None = None
    gate_layout: CovariateLayout | None = None
    miss: np.ndarray | None = None
    nan_cols: np.ndarray | None = None
    pattern_mode: bool = False
    overlap_trimmed: bool = False
    clusters: ClusterSupport | None = None
    diagnostic: tuple[np.ndarray, tuple[str, ...], tuple[str, ...]] | None = None

    @property
    def covariates(self) -> list[str]:
        """Column names of `X`, in order."""
        return list(self.layout.names)

    def indicator(self, a: int) -> np.ndarray:
        """1.0 on arm ``a``'s rows, 0.0 elsewhere."""
        return (self.arm == a).astype(float)

    def pair(self, a: int) -> slice | np.ndarray:
        """Rows of the treatment ``a`` / control comparison: every row of a
        binary cohort, otherwise a boolean mask."""
        if self.n_treatments == 1:
            return slice(None)
        return (self.arm == 0) | (self.arm == a)


@dataclass(slots=True)
class NuisancePredictions:
    """Cohort-aligned nuisance predictions carried through the overlap gate.

    `conditional[:, a - 1]` is treatment ``a``'s propensity against control
    alone; `marginal[:, a]` is arm ``a``'s coupled marginal propensity
    (control column 0). A ``"pair"`` outcome role is predicted on its
    comparison's rows only.
    """

    conditional: np.ndarray
    marginal: np.ndarray
    outcomes: dict[OutcomeRole, np.ndarray]


@dataclass(frozen=True, slots=True)
class ClusterSupport:
    """The rows a score can be nonzero on and the independent clusters they occupy.

    ``rows`` is ``slice(None)`` (every cohort row) or the supported rows'
    integer positions. A score that vanishes outside them is centered over,
    and reduced on, those rows alone: a cluster holding none of them carries
    an exactly zero total and none of the score's finite-sample information.
    Scores with terms on every cohort row -- augmented population terms and
    fitted-propensity estimating equations -- are supported on every row and
    reduce over every retained cluster, including clusters holding only other
    treatments. ``inv`` is each supported row's ordinal among the ``k``
    clusters those rows occupy; both are ``None`` without a declared cluster.
    These are independence groups, not a finite-sample residual design.
    """

    rows: slice | np.ndarray
    inv: np.ndarray | None
    k: int | None


# Every row, each its own independent unit.
_INDEPENDENT_ROWS = ClusterSupport(slice(None), None, None)


def _adjustment_columns(
    rows: nw.DataFrame, covariates: Sequence[str], cohort: AdjustmentCohort
) -> tuple[np.ndarray, CovariateLayout]:
    """One float matrix over the requested covariates and its layout.

    A numeric or boolean column is read as float with nulls as NaN; a
    string, categorical or enum column becomes level codes (NaN where
    null); an object column is whichever its non-null values all are. Any
    other dtype -- dates, times, nested values, mixed objects -- refuses by
    name: it has no adjustment meaning a learner could use."""
    columns: list[np.ndarray] = []
    levels: list[tuple[str, ...] | None] = []
    for name in covariates:
        dtype = rows.schema[name]
        values: list[object] | None = None
        if dtype.is_numeric() or dtype == nw.Boolean:
            kind = "numeric"
        elif dtype in (nw.String, nw.Categorical, nw.Enum):
            kind = "categorical"
        elif dtype == nw.Object:
            values = rows[name].to_list()
            kind = classify_objects(values)
        else:
            kind = None
        if kind is None:
            _raise(
                "estimation.adjust_common.covariate_dtype",
                method=cohort.method,
                covariate=name,
                dtype=str(dtype),
                metric=cohort.metric.name,
                contrast=_contrast_text(cohort.treatment_groups, cohort.control_group),
            )
        if kind == "numeric":
            if values is None:
                column = np.asarray(rows[name].cast(nw.Float64).to_numpy(), dtype=float)
            else:
                column = np.fromiter(
                    (math.nan if is_null(v) else float(cast(Any, v)) for v in values),
                    dtype=float,
                    count=len(values),
                )
            columns.append(column)
            levels.append(None)
            continue
        # `to_list()` keeps a null level null on every backend; a dictionary
        # array's `to_numpy()` does not.
        codes, labels = level_codes(values if values is not None else rows[name].to_list())
        columns.append(codes)
        levels.append(labels)
    layout = CovariateLayout(tuple(covariates), tuple(levels))
    return np.column_stack(columns), layout


def _prepare_adjustment_cohort(cohort: AdjustmentCohort) -> AdjustmentData:
    """Build the eligible cohort arrays: the control and every requested
    treatment, in source frame order."""
    lead = cohort.lead
    natives = (lead.control_native, *(request.treatment_native for request in cohort.requests))
    selected = nw.col("group_id") == natives[0]
    for native in natives[1:]:
        selected = selected | (nw.col("group_id") == native)
    rows = lead.frame.filter(selected)
    codes = {native: code for code, native in enumerate(natives)}
    groups = rows["group_id"].to_list()
    arm = np.fromiter(map(codes.__getitem__, groups), dtype=np.int64, count=len(groups))
    y = np.asarray(rows["y"].to_numpy(), dtype=float)
    unit_id = np.asarray(rows["unit_id"].to_list())
    X, layout = _adjustment_columns(rows, lead.covariates, cohort)
    labels = rows["cluster_id"].to_numpy() if lead.cluster is not None else None
    return AdjustmentData(arm, len(cohort.requests), y, X, unit_id, labels, layout)


def _resolve_adjustment_missingness(
    cohort: AdjustmentCohort, data: AdjustmentData
) -> AdjustmentData:
    """Apply the declared missing-data policy to the whole cohort before
    fitting or gating: one refusal, representation or retained set serves
    every comparison."""
    miss = ~np.isfinite(data.X)
    data.miss = miss
    if not miss.any():
        return data
    policy = cohort.design.adjustment.missing
    method = cohort.method
    metric_name = cohort.metric.name
    control_group = cohort.control_group
    treatment_groups = cohort.treatment_groups
    contrast = _contrast_text(treatment_groups, control_group)
    if policy == "refuse":
        raise _missing_covariate_refusal(
            miss,
            data.covariates,
            metric_name=metric_name,
            treatment_groups=treatment_groups,
            control_group=control_group,
            method=method,
        )
    if policy == "pattern":
        if method != "IPTW":
            outcome_models = "per-arm outcome models" if method == "AIPW" else "outcome model"
            raise IdentificationError(
                f"{method}: AdjustmentSet(missing='pattern') fits the propensity "
                f"separately per missingness pattern; per-pattern {outcome_models} "
                f"are not implemented for {method} (metric {metric_name!r}, "
                f"{contrast}). Declare "
                f"missing='impute-indicator' (pooled-mean impute plus a "
                f"missingness indicator in the adjustment set), declare "
                f"missing='allow' (NaN passed raw to explicitly supplied "
                f"NaN-native learners -- requires "
                f"Method(propensity_learner=..., outcome_learner=...)), or "
                f"use Method(name='iptw'), where missing='pattern' is "
                "supported.",
                code="adjust.identification.pattern_requires_iptw",
                context={
                    "method": method,
                    "metric_name": metric_name,
                    "treatment_groups": treatment_groups,
                    "control_group": control_group,
                },
            )
        data.pattern_mode = True
        return data
    if policy == "impute-indicator":
        data.X, data.layout, data.note = _impute_with_indicators(
            data.X,
            data.layout,
            miss,
            metric_name=metric_name,
            treatment_groups=treatment_groups,
            control_group=control_group,
            method=method,
        )
    elif policy == "complete-case":
        keep = ~miss.any(axis=1)
        n_total = data.arm.shape[0]
        n_kept = int(keep.sum())
        data.arm, data.y, data.X, data.unit_id = (
            data.arm[keep],
            data.y[keep],
            data.X[keep],
            data.unit_id[keep],
        )
        if data.labels is not None:
            data.labels = data.labels[keep]
        for code, empty_arm in enumerate(cohort.groups):
            if (data.arm == code).any():
                continue
            suffix = (
                "missing='impute-indicator' or missing='pattern' instead."
                if method == "IPTW"
                else "missing='impute-indicator' instead."
            )
            raise IdentificationError(
                f"missing='complete-case' emptied arm {empty_arm!r} for "
                f"metric {metric_name!r} ({contrast}) -- only {n_kept} of {n_total} "
                f"units are fully observed; keep every unit with {suffix}",
                code="adjust.identification.complete_case_emptied_arm",
                context={
                    "method": method,
                    "metric_name": metric_name,
                    "treatment_groups": treatment_groups,
                    "control_group": control_group,
                    "empty_arm": empty_arm,
                    "n_kept": n_kept,
                    "n_total": n_total,
                },
            )
        data.population = f"complete cases ({n_kept} of {n_total} units)"
    elif policy == "allow":
        data.allow_mode = True
        data.miss = _allow_nan_mask(
            data.X,
            data.covariates,
            metric_name=metric_name,
            treatment_groups=treatment_groups,
            control_group=control_group,
            method=method,
        )
        data.X_gate, data.gate_layout, _ = _impute_with_indicators(
            data.X,
            data.layout,
            data.miss,
            metric_name=metric_name,
            treatment_groups=treatment_groups,
            control_group=control_group,
            method=method,
        )
        data.note = _allow_note(data.miss, data.layout)
        data.nan_cols = data.miss.any(axis=0)
        # The gate representation above is imputed and never sees NaN; only
        # the learners' layout marks the columns whose NaN they receive.
        data.layout = data.layout.with_nullable(data.nan_cols)
    return data


def _outcome_what(role: OutcomeRole) -> str:
    """Prediction label named by nuisance refusals."""
    if role.kind == "pair":
        return "outcome"
    return "control-arm outcome" if role.arm == 0 else "treatment-arm outcome"


def _comparison_suffix(cohort: AdjustmentCohort, a: int) -> str:
    """Name the comparison a pair fit serves once several share the cohort."""
    if len(cohort.requests) == 1:
        return ""
    return f" for {_contrast_text(cohort.arm_groups(a), cohort.control_group)}"


def _outcome_fit_stage(cohort: AdjustmentCohort, role: OutcomeRole, j: int) -> str:
    """Arm fits name their arm; a pooled pair fit names its comparison only
    when several comparisons share the cohort."""
    if role.kind == "pair":
        return f"the fold-{j} outcome fit{_comparison_suffix(cohort, role.arm)}"
    if role.arm == 0:
        return f"the fold-{j} outcome fit for the control arm ({cohort.control_group!r})"
    return f"the fold-{j} outcome fit for the treatment arm ({cohort.groups[role.arm]!r})"


def _fit_predict(
    learner: Learner,
    X_train: np.ndarray,
    label: np.ndarray,
    X_test: np.ndarray,
    *,
    cohort: AdjustmentCohort,
    a: int,
    stage: str,
    what: str,
    allow_mode: bool,
) -> np.ndarray:
    """Fit one nuisance learner and predict `X_test`, refusing a mis-shaped
    prediction (and, under missing='allow', any learner raise)."""
    groups = cohort.arm_groups(a)
    if allow_mode:
        return _allow_fit_predict(
            learner,
            X_train,
            label,
            X_test,
            stage=stage,
            what=what,
            metric_name=cohort.metric.name,
            treatment_groups=groups,
            control_group=cohort.control_group,
            method=cohort.method,
        )
    learner.fit(X_train, label)
    return _prediction_shape_guard(
        learner.predict(X_test),
        expected_n=X_test.shape[0],
        learner_name=learner_name(learner),
        what=what,
        metric_name=cohort.metric.name,
        treatment_groups=groups,
        control_group=cohort.control_group,
        method=cohort.method,
    )


def _couple_propensities(conditional: np.ndarray) -> np.ndarray:
    """Marginal arm propensities from treatment-versus-control conditionals.

    Column ``a - 1`` of `conditional` is q_a = P(A=a | A in {0, a}, X), so
    q_a / (1 - q_a) = p_a / p_0 and p_0 = 1 / (1 + sum_a p_a / p_0). The
    odds are combined in log space. One treatment keeps the direct binary
    complement. A conditional of exactly 0 or 1 is not clipped: it yields a
    zero arm propensity, which the overlap gate reports; when two or more
    treatments saturate on one unit their marginal split is undetermined
    (NaN) while its control propensity is exactly zero.
    """
    n, n_treatments = conditional.shape
    marginal = np.empty((n, n_treatments + 1), order="F")
    if n_treatments == 1:
        q = conditional[:, 0]
        marginal[:, 0] = 1.0 - q
        marginal[:, 1] = q
        return marginal
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = np.log(conditional) - np.log1p(-conditional)
        log_total = np.logaddexp.reduce(log_odds, axis=1, initial=0.0)
        marginal[:, 0] = np.exp(-log_total)
        marginal[:, 1:] = np.exp(log_odds - log_total[:, None])
    saturated = log_odds == np.inf
    alone = saturated & (saturated.sum(axis=1) == 1)[:, None]
    treatments = marginal[:, 1:]
    treatments[alone] = 1.0
    return marginal


def _crossfit_splits(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    folds: int,
    arm_rows: list[np.ndarray],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """One cohort-wide, arm-stratified, cluster-atomic fold vector as
    `(fold, test, train)` masks; every training split must hold each
    treatment and the control."""
    method = cohort.method
    metric_name = cohort.metric.name
    control_group = cohort.control_group
    n = data.arm.shape[0]
    if n < 2 * folds:
        _raise(
            "estimation.adjust_common.needs_least_folds",
            method=method,
            needed=2 * folds,
            n=n,
            metric=metric_name,
            contrast=_contrast_text(cohort.treatment_groups, control_group),
        )
    if data.labels is not None:
        total_clusters = _cohort_clusters(data).k
        assert cohort.lead.cluster is not None and total_clusters is not None
        check_total_clusters(
            metric_name, cohort.lead.cluster, total_clusters, warn=False, stacklevel=4
        )
    fold = _fold_ids(data.unit_id, data.arm, folds, cluster_ids=data.labels)
    splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    for j in range(folds):
        test = fold == j
        if not test.any():
            continue
        train = ~test
        if not train.any():
            _raise(
                "estimation.adjust_common.fold_would_train",
                method=method,
                j=j,
                metric=metric_name,
                contrast=_contrast_text(cohort.treatment_groups, control_group),
            )
        for a in range(1, data.n_treatments + 1):
            if not (train & arm_rows[a]).any() or not (train & arm_rows[0]).any():
                _raise(
                    "estimation.adjust_common.fold_training_split",
                    method=method,
                    j=j,
                    metric=metric_name,
                    contrast=_contrast_text(cohort.arm_groups(a), control_group),
                )
        splits.append((j, test, train))
    return splits


def _probe_allow_learners(
    cohort: AdjustmentCohort, data: AdjustmentData, spec: NuisanceFoldSpec
) -> None:
    """Probe each nuisance factory once for NaN capability under missing='allow'.

    The probe is shaped like the design the supplied learner actually
    sees: a null-bearing categorical plants NaN in every one of its level
    columns."""
    assert data.nan_cols is not None
    probes: list[tuple[str, Callable[[], Learner]]] = [("propensity", spec.propensity_factory)]
    if spec.outcome_roles or spec.deferred_roles:
        probes.append(("outcome", spec.outcome_factory))
    for role_name, factory in probes:
        _probe_nan_capability(
            factory(),
            data.layout.encoded_width(),
            data.layout.expand_columns(data.nan_cols),
            role=role_name,
            metric_name=cohort.metric.name,
            treatment_groups=cohort.treatment_groups,
            control_group=cohort.control_group,
            method=cohort.method,
        )


def _guard_crossfit_predictions(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    conditional: np.ndarray,
    outcomes: dict[OutcomeRole, np.ndarray],
    learner_names: dict[OutcomeRole | int, str],
    pairs: list[np.ndarray],
) -> None:
    """Refuse non-finite nuisance predictions and out-of-range propensities.

    `learner_names` maps a treatment code to its propensity learner's name and
    each outcome role to its learner's name. A pair role is checked on its
    comparison's rows, the only rows it predicts."""
    metric_name, control_group, method = cohort.metric.name, cohort.control_group, cohort.method
    for a in range(1, data.n_treatments + 1):
        _finite_guard(
            conditional[:, a - 1],
            what="propensity",
            learner_name=learner_names[a],
            metric_name=metric_name,
            treatment_groups=cohort.arm_groups(a),
            control_group=control_group,
            method=method,
            allow_mode=data.allow_mode,
        )
    for role, values in outcomes.items():
        _finite_guard(
            values if role.kind == "arm" or not pairs else values[pairs[role.arm]],
            what=_outcome_what(role),
            learner_name=learner_names[role],
            metric_name=metric_name,
            treatment_groups=cohort.arm_groups(role.arm),
            control_group=control_group,
            method=method,
            allow_mode=data.allow_mode,
        )
    for a in range(1, data.n_treatments + 1):
        _propensity_range_guard(
            conditional[:, a - 1],
            learner_name=learner_names[a],
            metric_name=metric_name,
            treatment_groups=cohort.arm_groups(a),
            control_group=control_group,
            method=method,
        )


def _crossfit_nuisances(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    spec: NuisanceFoldSpec,
    folds: int,
) -> NuisancePredictions:
    """Cross-fit every treatment-versus-control propensity and `spec`'s
    outcome roles on one cohort-wide fold vector, then couple the conditional
    propensities into marginal arm propensities. Comparison rows train each
    comparison's models; every heldout cohort row receives each propensity
    prediction. Each model's level encoding is fitted on its own training
    rows, and the levels a heldout row carried into a fit that never saw
    them are disclosed once the stage is done. IPTW's in-sample propensity
    fits stay in `iptw.py`."""
    n = data.arm.shape[0]
    n_treatments = data.n_treatments
    binary = n_treatments == 1
    arm_rows = [data.arm == a for a in range(n_treatments + 1)]
    treated = [arm_rows[a].astype(float) for a in range(1, n_treatments + 1)]
    # Arm-indexed comparison masks; a binary cohort is its own comparison.
    pairs = (
        []
        if binary
        else [arm_rows[0], *(arm_rows[0] | arm_rows[a] for a in range(1, n_treatments + 1))]
    )
    splits = _crossfit_splits(cohort, data, folds, arm_rows)
    if data.allow_mode:
        _probe_allow_learners(cohort, data, spec)
    propensity_factory = encoded_factory(spec.propensity_factory, data.layout)
    outcome_factory = encoded_factory(spec.outcome_factory, data.layout)
    conditional = np.empty((n, n_treatments), order="F")
    outcomes = {role: np.zeros(n) for role in (*spec.outcome_roles, *spec.deferred_roles)}
    learner_names: dict[OutcomeRole | int, str] = {}
    unseen = UnseenLevels(n)

    def fit_outcomes(
        models: dict[OutcomeRole, Learner],
        j: int,
        test: np.ndarray,
        train: np.ndarray,
        X_test: np.ndarray,
    ) -> None:
        for role, model in models.items():
            if role.kind == "arm":
                rows, target, X_target = train & arm_rows[role.arm], test, X_test
            elif binary:
                rows, target, X_target = train, test, X_test
            else:
                rows, target = train & pairs[role.arm], test & pairs[role.arm]
                X_target = data.X[target]
            outcomes[role][target] = _fit_predict(
                model,
                data.X[rows],
                data.y[rows],
                X_target,
                cohort=cohort,
                a=role.arm,
                stage=_outcome_fit_stage(cohort, role, j),
                what=_outcome_what(role),
                allow_mode=data.allow_mode,
            )
            learner_names[role] = learner_name(model)
            unseen.record(model, X_target, target)

    for j, test, train in splits:
        propensities = [propensity_factory() for _ in range(n_treatments)]
        models = {role: outcome_factory() for role in spec.outcome_roles}
        X_test = data.X[test]
        for a, learner in enumerate(propensities, start=1):
            rows = train if binary else train & pairs[a]
            conditional[test, a - 1] = _fit_predict(
                learner,
                data.X[rows],
                treated[a - 1][rows],
                X_test,
                cohort=cohort,
                a=a,
                stage=f"the fold-{j} propensity fit{_comparison_suffix(cohort, a)}",
                what="propensity",
                allow_mode=data.allow_mode,
            )
            learner_names[a] = learner_name(learner)
            unseen.record(learner, X_test, test)
        fit_outcomes(models, j, test, train, X_test)
    if spec.deferred_roles:
        # Optional roles fit only after every base fit, on the saved folds.
        for j, test, train in splits:
            models = {role: outcome_factory() for role in spec.deferred_roles}
            fit_outcomes(models, j, test, train, data.X[test])
    _guard_crossfit_predictions(cohort, data, conditional, outcomes, learner_names, pairs)
    _disclose_unseen_levels(cohort, data, unseen)
    return NuisancePredictions(conditional, _couple_propensities(conditional), outcomes)


def _disclose_unseen_levels(
    cohort: AdjustmentCohort, data: AdjustmentData, unseen: UnseenLevels
) -> None:
    """Carry one fit stage's unseen levels into the cohort note and the
    coded advisory: rows a nuisance fit scored as its reference level
    because their level was absent from its training rows. Nothing is
    gated or trimmed; the disclosure is the diagnosis."""
    summary = unseen.summary()
    if not summary:
        return
    text = unseen_levels_text(summary, unseen.n)
    data.note = text if data.note is None else f"{data.note}; {text}"
    _warn(
        "estimation.adjust_common.unseen_level_advisory",
        method=cohort.method,
        metric_name=cohort.metric.name,
        treatment_groups=cohort.treatment_groups,
        control_group=cohort.control_group,
        unseen=summary,
        n=unseen.n,
        stacklevel=5,
    )


def _apply_overlap_gate(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    nuisance: NuisancePredictions,
) -> tuple[AdjustmentData, NuisancePredictions]:
    """Apply one inclusive all-arm overlap mask and rebuild the retained population.

    A unit is supported when every marginal arm propensity, control included,
    is at least `gate.min_propensity`; with one treatment that is the band
    [g, 1 - g] on its propensity. The same mask serves every comparison, so a
    trimmed cohort is one retained population for every row."""
    gate = cohort.design.gate
    g = gate.min_propensity
    marginal = nuisance.marginal
    if data.n_treatments == 1:
        e = marginal[:, 1]
        outside = (e < g) | (e > 1 - g)
    else:
        outside = ~(marginal >= g).all(axis=1)
    data.overlap_trimmed = bool(outside.any())
    if not data.overlap_trimmed:
        return data, nuisance
    groups = cohort.groups
    method = cohort.method
    metric_name = cohort.metric.name
    n_total = data.arm.shape[0]
    context_scope = {
        "method": method,
        "metric_name": metric_name,
        "treatment_groups": cohort.treatment_groups,
        "control_group": cohort.control_group,
    }
    if gate.overlap == "refuse":
        raise IdentificationError(
            _overlap_refusal_message(data.arm, marginal, outside, g, groups, method=method),
            code="adjust.identification.overlap_gate_refused",
            context={
                **context_scope,
                "n_outside": int(outside.sum()),
                "n_total": n_total,
                "min_propensity": g,
                "outside_by_arm": tuple(
                    (
                        group,
                        int((outside & (data.arm == code)).sum()),
                        int((data.arm == code).sum()),
                    )
                    for code, group in enumerate(groups)
                ),
                "propensity_ranges": _propensity_ranges(marginal, groups),
            },
        )
    kept = ~outside
    data.arm, data.y, data.X, data.unit_id = (
        data.arm[kept],
        data.y[kept],
        data.X[kept],
        data.unit_id[kept],
    )
    if data.labels is not None:
        data.labels = data.labels[kept]
    if data.clusters is not None:
        assert data.clusters.inv is not None and data.clusters.k is not None
        inv, k = _restrict_cluster_index(data.clusters.inv[kept], data.clusters.k)
        data.clusters = ClusterSupport(slice(None), inv, k)
    if data.X_gate is not None:
        data.X_gate = data.X_gate[kept]
    n_kept = int(kept.sum())
    band = (
        f"overlap e in [{g:g}, {1 - g:g}]"
        if data.n_treatments == 1
        else f"overlap with every marginal arm propensity >= {g:g}"
    )
    for code, empty_arm in enumerate(groups):
        if (data.arm == code).any():
            continue
        raise IdentificationError(
            f"trimming to {band} emptied arm {empty_arm!r} for metric {metric_name!r} "
            f"({_contrast_text(cohort.treatment_groups, cohort.control_group)}) -- only "
            f"{n_kept} of {n_total} units survived{_infeasible_band_note(len(groups), g)}; "
            "widen gate.min_propensity, add covariates that improve overlap, or "
            "use gate.overlap='refuse' to reject the comparison outright",
            code="adjust.identification.overlap_trim_emptied_arm",
            context={
                **context_scope,
                "empty_arm": empty_arm,
                "n_kept": n_kept,
                "n_total": n_total,
                "min_propensity": g,
            },
        )
    overlap_population = f"{band} ({n_kept} of {n_total} units)"
    data.population = (
        overlap_population
        if data.population is None
        else f"{data.population}; {overlap_population}"
    )
    return data, NuisancePredictions(
        nuisance.conditional[kept],
        marginal[kept],
        {role: values[kept] for role, values in nuisance.outcomes.items()},
    )


def _cohort_clusters(data: AdjustmentData) -> ClusterSupport:
    """Every retained cohort row and its cluster, including clusters holding
    only other treatment arms. The labels are sorted once per cohort; trimming
    restricts that index to the retained rows. Scores with terms on every
    cohort row reduce on it; `_validate_contrast` derives each comparison's
    own clusters from it."""
    if data.labels is None:
        return _INDEPENDENT_ROWS
    if data.clusters is None:
        inv, k = _cluster_index(data.labels)
        data.clusters = ClusterSupport(slice(None), inv, k)
    return data.clusters


def _diagnostic_design(data: AdjustmentData) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """The design balance diagnostics read, its column names and source
    covariates: the modal-reference level indicators over the retained
    cohort rows (a numeric cohort reads its own matrix). Under
    missing='allow' this is the imputed + indicator representation. Built
    once per cohort after the overlap gate, when the rows are final."""
    if data.diagnostic is None:
        if data.allow_mode:
            assert data.X_gate is not None and data.gate_layout is not None
            data.diagnostic = fixed_design(data.gate_layout, data.X_gate)
        else:
            data.diagnostic = fixed_design(data.layout, data.X)
    return data.diagnostic


def _validate_contrast(
    cohort: AdjustmentCohort,
    data: AdjustmentData,
    a: int,
    weights: np.ndarray,
    clusters: ClusterSupport,
) -> ClusterSupport:
    """Weighted covariate SMDs on one comparison's actual treatment/control
    rows (marginal arm weights), then its cluster support.

    A categorical covariate contributes one SMD row per non-reference level
    (``name=level``), the row its dummy column would have. Other treatment
    arms are never relabelled control: they stay out of this comparison's
    diagnostics, cluster count and support checks. The comparison's
    clusters are re-indexed from the cohort's (`clusters`); a binary cohort
    is its own comparison. The comparison needs two distinct clusters, two
    per arm when every cluster is arm-pure, and fewer than forty emit the
    small-cluster advisory. Returns the comparison's rows and their
    clusters, the support of any score that vanishes outside the
    comparison.
    """
    request = cohort.requests[a - 1]
    rows = data.pair(a)
    design, names, sources = _diagnostic_design(data)
    if data.allow_mode:
        # Interaction columns are formed on this comparison's rows only.
        X_gate = design[rows]
        d = data.indicator(a)[rows]
        arms = SmdArms.of(d == 1, d == 0, weights[rows])
        smds = [(name, _weighted_smd(X_gate[:, i], arms)) for i, name in enumerate(names)]
        smds += _allow_interaction_smds(X_gate, names, sources, data.covariates, arms)
    else:
        arms = SmdArms.of(data.arm == a, data.arm == 0, weights)
        smds = [(name, _weighted_smd(design[:, i], arms)) for i, name in enumerate(names)]
    gate = request.design.gate
    if gate.max_smd is not None:
        exceeded = [(name, smd) for name, smd in smds if abs(smd) > gate.max_smd]
        if exceeded:
            named = ", ".join(f"{name} (SMD={smd:.3f})" for name, smd in exceeded)
            detail = {
                "DML": "under inverse-propensity weighting. This is a propensity-model quality check (DML itself residualizes rather than weights) -- AdjustmentSet.covariates does not achieve balance under the fitted propensity for this comparison",
                "AIPW": "under inverse-propensity weighting. This is a propensity-model quality check (AIPW's per-arm outcome models are the second line of defense) -- AdjustmentSet.covariates does not achieve balance under the fitted propensity for this comparison",
                "IPTW": "after weighting -- AdjustmentSet.covariates does not achieve balance for this comparison",
            }[request.method]
            raise IdentificationError(
                f"{request.method} balance gate refused for metric {request.metric.name!r} "
                f"(contrast {request.treatment_group!r} vs {request.control_group!r}): {named} "
                f"exceed gate.max_smd={gate.max_smd:g} {detail}",
                code="adjust.identification.balance_gate_exceeded",
                context={
                    "method": request.method,
                    "metric_name": request.metric.name,
                    "treatment_group": request.treatment_group,
                    "control_group": request.control_group,
                    "exceeded": tuple(exceeded),
                    "max_smd": gate.max_smd,
                },
            )
    else:
        imbalanced = [(name, smd) for name, smd in smds if abs(smd) > 0.1]
        if imbalanced:
            named = ", ".join(f"{name} (SMD={smd:.3f})" for name, smd in imbalanced)
            if request.method == "DML":
                detail = (
                    "under inverse-propensity weighting (a propensity-model quality "
                    "check; DML itself residualizes rather than weights); set "
                    "gate.max_smd to enforce a hard refusal threshold"
                )
            elif request.method == "AIPW":
                detail = (
                    "under inverse-propensity weighting (AIPW's per-arm outcome models "
                    "are the second line of defense); set gate.max_smd to enforce a "
                    "hard refusal threshold"
                )
            else:
                detail = "after weighting; set gate.max_smd to enforce a hard refusal threshold"
            _warn(
                "estimation.adjust_common.covariate_balance_advisory",
                method=request.method,
                metric_name=request.metric.name,
                treatment_group=request.treatment_group,
                control_group=request.control_group,
                named=named,
                detail=detail,
                stacklevel=4,
            )
    if isinstance(rows, slice):
        support = clusters
    elif clusters.inv is None:
        support = ClusterSupport(np.flatnonzero(rows), None, None)
    else:
        assert clusters.k is not None
        support_rows = np.flatnonzero(rows)
        inv, k = _restrict_cluster_index(clusters.inv[support_rows], clusters.k)
        support = ClusterSupport(support_rows, inv, k)
    if support.inv is None:
        return support
    assert cohort.lead.cluster is not None and support.k is not None
    _validate_pair_cluster_support(support.inv, support.k, data.indicator(a)[rows])
    check_total_clusters(cohort.metric.name, cohort.lead.cluster, support.k, stacklevel=5)
    return support


def _score_stats(
    psi: np.ndarray,
    *,
    metric: str,
    contrast: str,
    normalizer: float | None = None,
    support: ClusterSupport | None = None,
) -> ScoreStats:
    """Collapse influence scores without overflowing their second moment.

    ``support`` names the rows ``psi`` can be nonzero on and their clusters
    (default: every row, unclustered): the sums skip the other, exactly
    zero, rows while ``n`` still counts every row, and the cluster reduction
    centers over and counts only the supported rows."""
    if support is None:
        support = _INDEPENDENT_ROWS
    full = np.asarray(psi, dtype=float)
    n = len(full)
    values = full[support.rows]
    scale = float(np.max(np.abs(values), initial=0.0))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    # Preserve a non-unit scale when the raw square sum would underflow or
    # overflow; otherwise ordinary values retain the historical representation.
    square = scale * scale
    if np.isfinite(square * max(n, 1)) and square >= np.finfo(float).tiny:
        scale = 1.0
    scaled = values if scale == 1.0 else values / scale
    # IEEE products, as Python's float multiply: an overflow stays a silent inf.
    with np.errstate(over="ignore", under="ignore"):
        squares = scaled * scaled
    sum_psi = math.fsum(memoryview(scaled))
    sum_psi2 = math.fsum(memoryview(squares))
    fields: dict[str, Any] = {}
    if support.inv is not None and support.k is not None:
        fields, _ = _clustered_fields(values, support.inv, support.k)
    return ScoreStats(
        metric=metric,
        contrast=contrast,
        n=n,
        sum_psi=sum_psi,
        sum_psi2=sum_psi2,
        sum_d_tilde2=normalizer,
        score_scale=scale,
        **fields,
    )


def _refuse_near_zero_adjustment_denominator(
    request: AdjustedContrastRequest,
    mu0: float,
    se_mu0: float,
    *,
    label: str,
) -> None:
    """Guard scalar informative-prior ratios against weak denominators.

    Prior-free joint inference deliberately retains weak denominators so
    Fieller inversion can represent disconnected or unbounded sets.
    """
    if request.prior is not None and abs(mu0) <= 4.0 * se_mu0:
        _raise(
            "estimation.adjust_common.statistically_indistinguishable_from",
            label=label,
            denominator=mu0,
            se=se_mu0,
            metric=request.metric.name,
            treatment=request.treatment_group,
            control=request.control_group,
            estimator=request.method,
        )


def _centered_totals(
    values: np.ndarray, index: np.ndarray | None, length: int | None
) -> tuple[list[int], Fraction]:
    """Exact per-cluster totals of scores centered over their own rows, or the
    centered scores themselves without a cluster index, as integers times one
    exact step. With ``values == integers * unit`` (`_dyadic_integers`), a
    total is ``(n * sum - count * whole) * unit / n``."""
    integers, unit = _dyadic_integers(values)
    n = len(integers)
    whole = sum(integers)
    step = unit / n
    if index is None:
        return [n * value - whole for value in integers], step
    assert length is not None
    sums = [0] * length
    for value, cluster in zip(integers, index.tolist(), strict=True):
        sums[cluster] += value
    counts = np.bincount(index, minlength=length).tolist()
    return [n * total - count * whole for total, count in zip(sums, counts, strict=True)], step


def _integer_dot(left: list[int], right: list[int]) -> int:
    """Exact inner product of two equally long integer sequences."""
    return sum(starmap(operator.mul, zip(left, right, strict=True)))


def _bessel(k: int | None) -> Fraction:
    """K/(K-1) for a score supported on K independent clusters; 1 unclustered."""
    if k is None:
        return Fraction(1)
    if k < 2:
        _overlap_raise("estimation.adjust_overlap.cluster_needs_two", k=k)
    return Fraction(k, k - 1)


_CROSS_FACTOR_BITS = 64


def _bessel_cross(k_a: int | None, k_c: int | None) -> Fraction:
    """sqrt(_bessel(k_a) * _bessel(k_c)), rounded down to a multiple of
    2**-64 unless both supports count the same clusters (then exact), so a
    covariance scaled by it never exceeds its two scaled deviations."""
    if k_a == k_c:
        return _bessel(k_a)
    product = _bessel(k_a) * _bessel(k_c)
    shifted = (product.numerator << (2 * _CROSS_FACTOR_BITS)) // product.denominator
    return Fraction(math.isqrt(shifted), 1 << _CROSS_FACTOR_BITS)


def _joint_reference_from_influences(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    numerator_point: float,
    denominator_point: float,
    supports: tuple[ClusterSupport, ClusterSupport] | None = None,
) -> tuple[JointContrastReference | None, RelativeUnavailableReason | None]:
    """Build a joint ratio reference from complete per-unit influences.

    ``supports`` gives each influence the rows it can be nonzero on and
    their clusters (default: every row, unclustered, for both); the
    denominator's is either every cohort row or the numerator's own. Each
    influence is centered over its own rows and summed within clusters in
    exact integer arithmetic; with Gram matrix S of those complete totals and
    c_j = K_j / (K_j - 1) for support j, the covariance is
    diag(sqrt c) S diag(sqrt c) / N**2. Each variance is then that
    influence's own clustered variance and the correlation that of the
    totals, which keeps the matrix positive semidefinite.
    """
    from increment.estimation.results import _joint_reference_from_exact

    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.ndim != 1 or denominator.ndim != 1 or numerator.shape != denominator.shape:
        return None, "joint_covariance_unrepresentable"
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        return None, "joint_covariance_unrepresentable"
    n = len(numerator)
    if n == 0:
        return None, "joint_covariance_unrepresentable"
    num_support, den_support = (
        (_INDEPENDENT_ROWS, _INDEPENDENT_ROWS) if supports is None else supports
    )
    totals_c, step_c = _centered_totals(
        denominator[den_support.rows], den_support.inv, den_support.k
    )
    values_a = numerator[num_support.rows]
    if isinstance(den_support.rows, slice):
        # Every numerator row lies in the whole-cohort denominator's support:
        # total the numerator in the denominator's clusters (or rows).
        if den_support.inv is None:
            totals_a, step_a = _centered_totals(values_a, None, None)
            aligned_c = (
                totals_c
                if isinstance(num_support.rows, slice)
                else [totals_c[i] for i in num_support.rows.tolist()]
            )
        else:
            totals_a, step_a = _centered_totals(
                values_a, den_support.inv[num_support.rows], den_support.k
            )
            aligned_c = totals_c
    else:
        # A denominator on one comparison's rows shares the numerator's support.
        assert num_support is den_support
        totals_a, step_a = _centered_totals(values_a, num_support.inv, num_support.k)
        aligned_c = totals_c
    squared_n = Fraction(n) * Fraction(n)
    var_a = _bessel(num_support.k) * _integer_dot(totals_a, totals_a) * step_a * step_a / squared_n
    var_c = _bessel(den_support.k) * _integer_dot(totals_c, totals_c) * step_c * step_c / squared_n
    cov_ac = (
        _bessel_cross(num_support.k, den_support.k)
        * _integer_dot(totals_a, aligned_c)
        * step_a
        * step_c
        / squared_n
    )
    return _joint_reference_from_exact(
        a=numerator_point,
        c=denominator_point,
        var_a=var_a,
        var_c=var_c,
        cov_ac=cov_ac,
    )


def _infer_adjusted_contrast(
    request: AdjustedContrastRequest,
    *,
    point: float | None,
    scores: ScoreStats,
    population: str | None,
    abs_diff: float | None,
    abs_se: float | None,
    estimand: Literal["ate", "plr_slope", "overlap_subpopulation_ate"],
    note: str | None,
    n_clusters: int | None,
    dof: float | None,
    abs_dof: float | None = None,
    joint_result: tuple[JointContrastReference | None, RelativeUnavailableReason | None] = (
        None,
        None,
    ),
) -> LiftEstimate:
    """Attach inference metadata and value-scale notes to a contrast."""
    estimate = infer_ate(
        metric=request.metric.name,
        group_id=request.treatment_group,
        method=request.method.lower(),
        method_role="decision",
        point=point,
        scores=scores,
        prior=request.prior,
        alpha=request.alpha,
        alternative=request.alternative,
        population=population,
        null_lift=request.null_lift,
        null_abs=request.null_abs,
        preferred_direction=request.preferred_direction,
        abs_diff=abs_diff,
        abs_se=abs_se,
        value_scale=request.value_scale,
        dof=dof,
        abs_dof=abs_dof,
        n_clusters=n_clusters,
        joint_reference=joint_result[0],
        relative_unavailable_reason=joint_result[1],
        estimand=estimand,
    )
    if n_clusters is not None:
        covariance_note = (
            "Independent-cluster superpopulation score sandwich; asymptotic Normal "
            "reference. Bessel scaling does not correct fitted nuisance leverage; "
            "small-cluster calibration is unresolved."
        )
        note = f"{note}; {covariance_note}" if note else covariance_note
    note = _compose_absolute_note(note, request.metric.name, request.value_scale)
    return estimate.model_copy(update={"note": note}) if note is not None else estimate
