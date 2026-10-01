"""Honest-split validation of a fitted CATE model.

Ranking units by their own in-sample predicted effect and reporting the
top group's effect is a fabrication machine: with no heterogeneity at
all the top quintile still reads several times the true average effect.
This module keeps that number honest by content-hashing units into two
halves - half A fits the model and is discarded; half B, never seen by
the fit, supplies every reported number: sorted group effects (GATES,
quantile-cut by predicted effect), rank tests (AUTOC/Qini - one-sided,
does the ranking carry signal at all), and CLAN (who is in the most-
versus least-affected group).

``CateValidation.passed`` is ``autoc.p_value < alpha``; nothing
downstream may report a discovered-subgroup number while it is false.
:func:`targeting_rule_arrays` deploys a pre-committed top-*fraction*
cut over the held-out score, reported only when the gate passes -
reading the group table first and picking the cut costs 60-120%
upward bias, so *fraction* has no default.
:func:`select_targeting_rule_arrays` chooses among a fraction grid via
an inner/outer split instead of relaxing that pre-commitment.

Results are on the absolute outcome scale, matching
:mod:`increment.estimation.cate`. Rank tests follow Yadlowsky, Fleming,
Shah, Brunskill and Wager (2021); sorted-group and CLAN reports follow
Chernozhukov, Demirer, Duflo and Fernandez-Val (2018).

Randomized assignment uses centered inverse-propensity scores. Observational
comparisons can instead supply cross-fitted doubly robust scores, subject to
their declared overlap policy.
"""

from __future__ import annotations

import functools
import math
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, cast, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm
from scipy.stats import t as student_t

from increment._identity import canonical_id_strings
from increment.errors import (
    CodedModel,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.estimation._adjust.encoding import (
    CovariateLayout,
    FittedEncoding,
    UnseenLevels,
    classify_objects,
    encoded_factory,
    is_null,
    matrix_from_columns,
    unseen_levels_text,
)
from increment.estimation._adjust.overlap import IdentificationError
from increment.estimation._deployment import (
    Deployment,
    cluster_prefix,
    resolve_deploy_grain,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.cate import (
    CATE_OUTCOME_NULLS_NON,
    CATE_TREATMENT_BINARY,
    CateResult,
    CateScoreState,
    Covariate,
    _column,
    _dedup,
    _validate_cluster_weight,
    fit_cate,
)
from increment.estimation.crossfit import (
    N_FOLDS_LEAST,
    check_cluster_atomic,
    fold_assignments,
    outer_split,
    select_out_of_fold,
)
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.results import Estimate

if TYPE_CHECKING:
    from increment.estimation._adjust.learners import Learner
    from increment.semantics.design import IdentificationGate

# Built-in scores also accept the unencoded, plain-array nuisance design.
_ArrayPsiFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray],
]


class ScoreDesign(np.ndarray):
    """The float design a caller-supplied score reads, with its column names.

    Numeric adjustment columns pass through as they are; each categorical
    column is replaced by the 0/1 level indicators of one basis fitted on
    the workflow's training rows alone -- the modal training level is the
    reference, every other training level gets one column in descending
    training frequency, and a scored level the training rows never carried
    reads as the reference. One workflow hands every call the same basis,
    so a level means the same columns in each of them. ``columns`` names
    the columns (``name`` for a numeric column, ``name=level`` for an
    indicator) and ``sources`` the adjustment covariate each derives from.
    Row-only indexing and copies retain the names. Other views clear them,
    since a changed axis cannot inherit the original column meanings.
    """

    columns: tuple[str, ...]
    sources: tuple[str, ...]

    def __new__(
        cls, matrix: np.ndarray, columns: tuple[str, ...], sources: tuple[str, ...]
    ) -> ScoreDesign:
        design = np.asarray(matrix, dtype=float).view(type=cls)
        design.columns = columns
        design.sources = sources
        return design

    def __array_finalize__(self, _obj: object) -> None:
        self.columns, self.sources = (), ()

    def __getitem__(self, key: Any) -> Any:
        result = super().__getitem__(key)
        parts = key if isinstance(key, tuple) else (key,)
        rows_only = self.ndim == 2 and (
            len(parts) == 1
            or (
                len(parts) == 2
                and (
                    parts[1] is Ellipsis
                    or (
                        isinstance(parts[1], slice)
                        and parts[1].indices(self.shape[1]) == (0, self.shape[1], 1)
                    )
                )
            )
        )
        if isinstance(result, ScoreDesign) and result.ndim == 2 and rows_only:
            result.columns, result.sources = self.columns, self.sources
        return result

    def copy(self, order: Any = "C") -> ScoreDesign:
        return ScoreDesign(self.view(np.ndarray).copy(order=order), self.columns, self.sources)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        out: tuple[object, ...] | None = None,
        **kwargs: object,
    ) -> object:
        plain = [x.view(np.ndarray) if isinstance(x, ScoreDesign) else x for x in inputs]
        if out is not None:
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, ScoreDesign) else x for x in out
            )
        return getattr(ufunc, method)(*plain, **kwargs)


PsiFn = Callable[
    [np.ndarray, np.ndarray, ScoreDesign, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray],
]


# CDDF (2018) medians GATES/CLAN/rank-test numbers over many train/holdout
# splits for stability; this module reads every number off one deterministic
# split (see `_holdout_mask`) and does not implement that reduction.
_SINGLE_SPLIT_CAVEAT = (
    "every number here is read off one deterministic train/holdout split "
    "(crc32 content hash of the assignment identity -- cluster ids when "
    "supplied, otherwise unit ids); CDDF (2018) medians GATES/CLAN/rank "
    "numbers over many splits for stability, which this module does not do "
    "-- a different split can move the group effects, CLAN profile, and "
    "rank tests, though not the honest-split guarantee itself."
)

_SEEDED_STRATIFIED_SPLIT_CAVEAT = (
    "every number here is read off one seeded, treatment-stratified outer "
    "train/holdout split; CDDF (2018) medians GATES/CLAN/rank numbers over "
    "many splits for stability, which this module does not do -- a different "
    "seed can move the group effects, CLAN profile, and rank tests, though not "
    "the honest-split guarantee itself."
)


class ClusterSupportFailure(BaseModel):
    """Source-support failures, counted separately for each fold/arm and stage.

    ``count`` is one for an original-population failure and the number of
    affected replicates for bootstrap failures. A replicate may fail several
    checks; excluded replicates are B minus bootstrap_valid_repetitions.
    """

    model_config = ConfigDict(frozen=True)

    reason: str
    stage: Literal["original", "bootstrap"]
    fold: int | None = None
    arm: int | None = None
    count: int = Field(default=1, gt=0)


class GroupEffect(BaseModel):
    """The treatment effect inside one predicted-effect group of the holdout.

    ``group`` is 1-indexed, 1 the lowest predicted effect. ``effect`` and
    ``se`` summarize either raw arm outcomes or per-unit effect scores,
    according to the identification design used by the caller. With clusters,
    ``se`` and intervals include empirical-cutoff uncertainty via bootstrap;
    ``covariance_method`` identifies the conditional smooth contrast. Missing
    numbers carry an ``unavailable_reason`` code.
    """

    model_config = ConfigDict(frozen=True)

    group: int
    n: int
    mean_score: float | None
    effect: float | None
    se: float | None
    lb: float | None
    ub: float | None
    n_clusters: int | None = None
    cluster_weight: Literal["member_count", "equal"] | None = None
    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()


class RankTest(BaseModel):
    """A rank-weighted average treatment effect and its one-sided test.

    ``p_value`` tests H1: ``estimate > 0`` - the ranking beats treating
    everybody. The two-sided alternative is not interesting here, since
    a ranking that is reliably backwards is still a ranking nobody deploys.
    Clustered intervals enclose bootstrap-t and complete cluster jackknife-t
    intervals; tests require both to reject. Resampling provenance is recorded.
    """

    model_config = ConfigDict(frozen=True)

    estimate: float | None
    se: float | None
    p_value: float | None
    lb: float | None = None
    ub: float | None = None
    n_clusters: int | None = None
    cluster_weight: Literal["member_count", "equal"] | None = None
    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()


class ClanRow(BaseModel):
    """One covariate's profile across the most- and least-affected groups.

    Categorical covariates contribute one row per level, named
    ``"covariate=level"``, carrying that level's share of the group.
    """

    model_config = ConfigDict(frozen=True)

    covariate: str
    mean_most: float | None
    mean_least: float | None
    diff: float | None
    se: float | None
    lb: float | None
    ub: float | None
    n_clusters: int | None = None
    cluster_weight: Literal["member_count", "equal"] | None = None
    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()


class CateEvaluationPopulation(CodedModel, BaseModel):
    """Immutable roster and weighting provenance for a reported evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit_ids: tuple[str, ...] = Field(min_length=1)
    cluster_ids: tuple[str, ...] | None = None
    base_weights: tuple[float, ...] = Field(min_length=1)
    weighting: Literal["member_count", "equal"]
    split: Literal["honest", "outer"]
    seed: int | None = Field(default=None, ge=0, strict=True)
    retention: Literal["all", "overlap_trimmed"] = "all"
    overlap: Literal["overlap_subpopulation"] | None = None
    # Frozen observational nuisances, aligned to ``unit_ids``.  This is
    # producer-owned diagnostic state; it is never refit by a consumer.
    nuisance_predictions: tuple[tuple[float, ...], ...] | None = None
    score_method: Literal["centered_ipw", "frozen_dr", "custom"] | None = None

    @model_validator(mode="after")
    def _aligned(self) -> CateEvaluationPopulation:
        n = len(self.unit_ids)
        if (self.score_method == "frozen_dr" and self.nuisance_predictions is None) or (
            self.score_method in ("centered_ipw", "custom")
            and self.nuisance_predictions is not None
        ):
            _raise(
                "estimation.targeting.evaluation_population_invalid",
                reason="score method disagrees with frozen nuisances",
            )
        if len(self.base_weights) != n or (
            self.cluster_ids is not None and len(self.cluster_ids) != n
        ):
            _raise("estimation.targeting.evaluation_population_invalid", reason="unaligned fields")
        if self.nuisance_predictions is not None:
            if len(self.nuisance_predictions) != 3 or any(
                len(values) != n for values in self.nuisance_predictions
            ):
                _raise(
                    "estimation.targeting.evaluation_population_invalid",
                    reason="unaligned nuisance predictions",
                )
            if not all(
                math.isfinite(value) for values in self.nuisance_predictions for value in values
            ):
                _raise(
                    "estimation.targeting.evaluation_population_invalid",
                    reason="nonfinite nuisance predictions",
                )
            if any(not 0 < value < 1 for value in self.nuisance_predictions[0]):
                _raise(
                    "estimation.targeting.evaluation_population_invalid",
                    reason="frozen nuisance propensities must have overlap",
                )
        if len(set(self.unit_ids)) != n:
            _raise(
                "estimation.targeting.evaluation_population_invalid", reason="duplicate unit IDs"
            )
        if not all(math.isfinite(w) and w > 0 for w in self.base_weights):
            _raise("estimation.targeting.evaluation_population_invalid", reason="invalid weights")
        if self.weighting == "equal" and self.cluster_ids is None:
            _raise(
                "estimation.targeting.evaluation_population_invalid",
                reason="missing cluster IDs",
            )
        if (self.split == "outer") != (self.seed is not None):
            _raise(
                "estimation.targeting.evaluation_population_invalid", reason="split seed mismatch"
            )
        if (self.retention == "overlap_trimmed") != (self.overlap is not None):
            _raise("estimation.targeting.evaluation_population_invalid", reason="overlap mismatch")
        expected = (
            np.ones(n)
            if self.cluster_ids is None
            else _target_weights(np.asarray(self.cluster_ids), self.weighting)
        )
        if any(
            actual != expected_value
            for actual, expected_value in zip(self.base_weights, expected, strict=True)
        ):
            _raise(
                "estimation.targeting.evaluation_population_invalid",
                reason="weights disagree with weighting",
            )
        return self


def _evaluation_population(
    holdout: _Holdout,
    *,
    split: Literal["honest", "outer"],
    seed: int | None,
    retention: Literal["all", "overlap_trimmed"],
    overlap: Literal["overlap_subpopulation"] | None,
) -> CateEvaluationPopulation:
    ids = holdout.cluster_ids
    weights = (
        np.ones(holdout.unit_ids.size, dtype=float)
        if ids is None
        else _target_weights(ids, holdout.cluster_weight)
    )
    frozen = getattr(holdout.psi_fn, "keywords", {}).get("frozen_predictions")
    nuisance = (
        None
        if frozen is None
        else tuple(tuple(float(value) for value in values) for values in frozen)
    )
    return CateEvaluationPopulation(
        unit_ids=tuple(str(value) for value in holdout.unit_ids),
        cluster_ids=None if ids is None else tuple(str(value) for value in ids),
        base_weights=tuple(float(value) for value in weights),
        weighting=holdout.cluster_weight,
        split=split,
        seed=seed,
        retention=retention,
        overlap=overlap,
        nuisance_predictions=nuisance,
        score_method=(
            "centered_ipw"
            if holdout.psi_fn is _ipw_psi_fn
            else "frozen_dr"
            if nuisance is not None
            else "custom"
        ),
    )


class CateValidation(CodedModel, BaseModel):
    """Everything the held-out half says about a fitted CATE model.

    ``passed`` is ``autoc.p_value < alpha``; while false, the only
    defensible number is the average effect, ``holdout_ate`` - the
    effect on the same rows the groups and rank tests use. Randomized
    sources use a difference in means with a Welch SE; observational
    sources use the mean cross-fitted doubly robust score with its SE.

    ``groups``/``clan`` intervals are Bonferroni-corrected across their
    own family (``n_groups`` group intervals, ``len(clan)`` CLAN rows):
    a reader scanning every row for the one excluding zero pays the true
    familywise error, not the per-row nominal ``alpha`` (CDDF 2018).
    ``split_caveat`` names the single-split limitation this correction
    does not address. For declared clusters, top-level ``uncertainty_method``,
    ``reference_df``, and ``unavailable_reason`` describe the holdout ATE;
    each group/rank/CLAN row records its own bootstrap uncertainty. A missing
    AUTOC p-value closes the gate. Nuisances are frozen using training rows.
    """

    model_config = ConfigDict(frozen=True)

    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()
    cluster_weight: Literal["member_count", "equal"] | None = None
    holdout_ate_se: float | None = None

    n_train: int
    n_holdout: int
    holdout_ate: Estimate | None
    groups: tuple[GroupEffect, ...]
    autoc: RankTest
    qini: RankTest
    clan: tuple[ClanRow, ...]
    alpha: float
    passed: bool
    population: str | None = None
    evaluation_population: CateEvaluationPopulation | None = None
    split_caveat: str = _SINGLE_SPLIT_CAVEAT
    #: Distinct declared clusters in the holdout population; `None` when the
    #: source declares no cluster (`_Holdout.cluster_ids` is `None`).
    n_clusters: int | None = Field(default=None, strict=True, gt=0)

    @model_validator(mode="after")
    def _validate_cluster_count(self) -> CateValidation:
        if self.n_clusters is not None and self.n_clusters > self.n_holdout:
            _raise(
                "estimation.targeting.cluster_count_bounds",
                n_clusters=self.n_clusters,
                n_holdout=self.n_holdout,
            )
        snapshot = self.evaluation_population
        if snapshot is not None:
            count = None if snapshot.cluster_ids is None else len(set(snapshot.cluster_ids))
            if (
                len(snapshot.unit_ids) != self.n_holdout
                or count != self.n_clusters
                or snapshot.weighting != (self.cluster_weight or "member_count")
                or snapshot.overlap != self.population
            ):
                _raise(
                    "estimation.targeting.evaluation_population_invalid",
                    reason="snapshot disagrees with validation population",
                )
        return self


class TargetingRule(CodedModel, BaseModel):
    """A frozen deployment candidate and its honest-split evidence gate.

    ``fraction`` is the requested budget; ``achieved_fraction`` is its realized
    holdout share under ``cluster_weight``. Cluster policies pool scores, break
    ties by canonical ID and take the longest feasible whole-cluster prefix.
    Their threshold is descriptive, never a substitute for the prefix rule.
    Unit policies retain their frozen score cutoff.

    ``predict`` applies this candidate to new columns without fitting again.
    ``recommendation`` is ``"target"`` only when the evidence gate passes;
    otherwise it recommends treating everyone alike based on the average effect.
    A passing, nonempty candidate reports its conditional policy effect and
    uplift as points only: this same holdout gates and reports them, so nominal
    intervals would be invalid after selection. Empty candidates retain null
    effects and their exact unavailable reason, rather than fabricated zeros.
    """

    model_config = ConfigDict(frozen=True)

    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()
    cluster_weight: Literal["member_count", "equal"] = "member_count"

    fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    recommendation: Literal["target", "simple"]
    validation: CateValidation
    population: str | None = None
    threshold: float | None
    policy_value: Estimate | None
    uplift_vs_average: Estimate | None
    #: Mirrors `validation.n_clusters` at the policy's own top level -- the
    #: honest-split cluster count a caller deploying this rule should know
    #: without reaching into the nested validation.
    n_clusters: int | None = Field(default=None, strict=True, gt=0)
    intervention_grain: Literal["unit", "cluster"] = "unit"
    deploy_grain: Literal["unit", "cluster"]
    budget_rule: Literal["unit_threshold", "whole_cluster_prefix"]
    achieved_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    score_state: CateScoreState
    score_cutoff: float | None = Field(allow_inf_nan=False)
    #: Declared model/adjustment columns; deployment identity is separate metadata.
    required_columns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_cluster_metadata(self) -> TargetingRule:
        if self.n_clusters != self.validation.n_clusters:
            _raise(
                "estimation.targeting.cluster_count_mismatch",
                n_clusters=self.n_clusters,
                validation_n_clusters=self.validation.n_clusters,
            )
        if self.intervention_grain == "cluster" and self.n_clusters is None:
            _raise("estimation.targeting.cluster_grain_count_required")
        resolve_deploy_grain(
            self.intervention_grain, self.deploy_grain, clustered=self.n_clusters is not None
        )
        _validate_cluster_weight(self.cluster_weight, clustered=self.n_clusters is not None)
        expected = "whole_cluster_prefix" if self.deploy_grain == "cluster" else "unit_threshold"
        if self.deploy_grain == "cluster" and self.achieved_fraction > self.fraction:
            _raise("estimation.targeting.policy_metadata")
        if self.budget_rule != expected or (
            self.deploy_grain == "unit" and 0 < self.fraction < 1 and self.score_cutoff is None
        ):
            _raise("estimation.targeting.policy_metadata")
        threshold_required = self.recommendation == "target" and self.achieved_fraction > 0
        if (threshold_required and self.threshold is None) or (
            self.threshold is not None and self.threshold != self.score_cutoff
        ):
            _raise("estimation.targeting.policy_metadata")
        if self.cluster_weight != (self.validation.cluster_weight or "member_count"):
            _raise("estimation.targeting.policy_metadata")
        return self

    def predict(
        self,
        cols: Mapping[str, np.ndarray],
        *,
        cluster_ids: np.ndarray | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
    ) -> np.ndarray:
        """Candidate policy actions aligned to rows, using the saved fitted score.

        ``recommendation`` remains the evidence gate for deploying this candidate.
        Unit policies use the frozen cutoff; cluster policies recompute their
        whole-cluster prefix budget on this batch and broadcast member actions.
        """
        grain = resolve_deploy_grain(
            self.intervention_grain,
            self.deploy_grain if deploy_grain is None else deploy_grain,
            clustered=cluster_ids is not None,
        )
        if grain != self.deploy_grain:
            _raise("estimation.targeting.policy_grain_mismatch")
        score, ids = self.score_state._score_for_roster(cols, cluster_ids)
        if grain == "cluster":
            assert ids is not None
            return cluster_prefix(score, ids, self.fraction, self.cluster_weight).targeted
        if self.fraction in (0, 1):
            return np.full(score.size, self.fraction == 1, dtype=bool)
        assert self.score_cutoff is not None
        return score >= self.score_cutoff


class FractionScore(BaseModel):
    """One grid fraction's pooled out-of-fold net benefit on the inner half.

    The selected fraction's row (``fraction == selected_fraction`` on the
    enclosing :class:`TargetingSelection`) ships ``net_benefit`` without
    ``lb``/``ub``/``level`` when it was chosen among multiple candidate
    fractions: it is the argmax of several noisy per-fraction estimates, so
    its own interval is winner's-curse inflated and not honest at any
    nominal level - the outer half's ``rule`` is the reportable verdict for
    the winner. A singleton grid has no data-dependent selection to
    correct for, so its one row keeps its interval. Every other row keeps
    its interval.
    """

    model_config = ConfigDict(frozen=True)

    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()
    n_clusters: int | None = None
    cluster_weight: Literal["member_count", "equal"] | None = None

    fraction: float
    achieved_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    net_benefit: Estimate
    n: int


class TargetingSelection(CodedModel, BaseModel):
    """A fraction chosen honestly, and the locked rule's untouched-test verdict.

    ``inner`` is the selection table: each grid fraction's net benefit
    ``E[1{targeted}(tau - cost)]`` on inner units scored by models that
    never saw them, using IPW contributions for randomized sources and
    cross-fitted doubly robust scores for observational sources.
    ``selected_fraction`` is its argmax (ties to the
    smaller fraction), locked before the outer test is read. ``rule``
    is the ordinary :class:`TargetingRule` evaluated once on the outer
    half. ``population`` records when overlap trimming restricts the inner
    selection and training population; it is independent of ``rule.population``,
    which records trimming on the outer test. ``seed`` and the grid are
    pre-commitments: rerunning with a new seed until the answer improves is
    the failure mode this workflow prevents.
    Clustered bootstrap availability and valid-repetition metadata mirror the
    selected inner row; outer-test uncertainty remains on ``rule``.
    """

    model_config = ConfigDict(frozen=True)

    uncertainty_method: str | None = None
    covariance_method: str | None = None
    reference_df: float | None = None
    unavailable_reason: str | None = None
    bootstrap_seed: int | None = None
    bootstrap_repetitions: int | None = None
    bootstrap_valid_repetitions: int | None = None
    support_failures: tuple[ClusterSupportFailure, ...] = ()
    n_clusters: int | None = None
    cluster_weight: Literal["member_count", "equal"] | None = None

    fractions: tuple[float, ...]
    inner: tuple[FractionScore, ...]
    selected_fraction: float
    cost_per_treated: float
    n_folds: int
    seed: int
    rule: TargetingRule
    population: str | None = None

    @model_validator(mode="after")
    def _policy_contract(self) -> TargetingSelection:
        snapshot = self.rule.validation.evaluation_population
        if (
            self.selected_fraction not in self.fractions
            or self.rule.fraction != self.selected_fraction
            or tuple(row.fraction for row in self.inner) != self.fractions
            or self.rule.cluster_weight != (self.cluster_weight or "member_count")
            or (snapshot is not None and (snapshot.split != "outer" or snapshot.seed != self.seed))
        ):
            _raise("estimation.targeting.policy_metadata")
        return self

    @property
    def deploy_grain(self) -> Literal["unit", "cluster"]:
        return self.rule.deploy_grain

    @property
    def intervention_grain(self) -> Literal["unit", "cluster"]:
        return self.rule.intervention_grain

    @property
    def budget_rule(self) -> Literal["unit_threshold", "whole_cluster_prefix"]:
        return self.rule.budget_rule

    @property
    def achieved_fraction(self) -> float:
        return self.rule.achieved_fraction


def _holdout_mask(unit_ids: np.ndarray, cluster_ids: np.ndarray | None = None) -> np.ndarray:
    """Held-out half of the units: ``crc32(unit_id) % 2 == 1``.

    Keyed on unit id content, so the split is invariant to row order
    and process (unlike Python's salted ``hash``); not random, since
    the honest-split argument only needs the two halves independent of
    (Y, D, X). ``unit_ids`` must be string-typed - hashing a numeric
    dtype's ``str()`` is not stable across dtypes for the same id.

    When *cluster_ids* is supplied, the same content hash keys on the
    cluster id instead and is broadcast to every member: a cluster
    never lands on both sides of this default split.
    """
    if unit_ids.dtype.kind not in ("U", "S", "O"):
        _raise("estimation.targeting.unit_ids_string", unit_ids=unit_ids.dtype)
    if cluster_ids is None:
        keys = unit_ids
    else:
        keys = cluster_ids
    return np.array([zlib.crc32(str(u).encode()) % 2 == 1 for u in keys], dtype=bool)


def _mean(x: np.ndarray) -> float:
    """Sample mean, NaN on an empty group rather than a numpy warning."""
    return float(x.mean()) if x.size else math.nan


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Difference in means ``mean(a) - mean(b)`` and its Welch standard error.

    Both NaN unless each side holds at least two units: one unit
    carries no within-group variance, so there is no SE to report.
    """
    if a.size < 2 or b.size < 2:
        return math.nan, math.nan
    diff = float(a.mean()) - float(b.mean())
    se = math.sqrt(float(a.var(ddof=1)) / a.size + float(b.var(ddof=1)) / b.size)
    return diff, se


def _ipw_psi(y: np.ndarray, d: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Per-unit scores for the conditional effect: ``psi_i`` tracks ``tau(x_i)``.

    ``psi_i = (y_i - ybar)(d_i - pbar) / (pbar (1 - pbar))``, ``pbar``
    the treated share of these rows. The uncentered transform is exactly
    conditionally unbiased, ``E[psi | x_i] = tau(x_i)``; centering
    ``y`` buys a large variance reduction for an O(1/n) bias::

        E[psi_i | x] = ((n - 2) tau(x_i) + taubar) / (n - 1)

    i.e. ``tau(x_i)`` pulled toward the average effect by ``1/(n-1)`` -
    a strictly increasing affine map, so ordering (all the rank tests
    read) is exactly preserved. The prognostic mean ``mu(x)`` drops out
    entirely; the trade is worth it at any realistic n, since removing
    it cuts rank-test noise several-fold against a bias shrinking like
    ``1/n`` on a statistic whose own SE shrinks like ``1/sqrt(n)``.
    """
    if weights is not None:
        pbar = float(np.average(d, weights=weights))
        centered = y - y[0]
        residual = centered - float(np.average(centered, weights=weights))
        return residual * (d - pbar) / (pbar * (1 - pbar))
    pbar = float(d.mean())
    return (y - float(y.mean())) * (d - pbar) / (pbar * (1.0 - pbar))


def _ipw_psi_fn(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    unit_ids: np.ndarray,
    cluster_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized-assignment score and a mask retaining every unit."""
    del X, unit_ids, cluster_ids
    return _ipw_psi(y, d), np.ones(y.shape, dtype=bool)


def _dr_factories(
    propensity_learner: Callable[[], Learner],
    outcome_learner: Callable[[], Learner],
    layout: CovariateLayout | None,
) -> tuple[Callable[[], Learner], Callable[[], Learner]]:
    """The nuisance factories as supplied, or wrapped to fit the level
    encoding of *layout* on each learner's own training rows."""
    if layout is None:
        return propensity_learner, outcome_learner
    return encoded_factory(propensity_learner, layout), encoded_factory(outcome_learner, layout)


def _dr_psi(
    y: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    unit_ids: np.ndarray,
    cluster_ids: np.ndarray | None,
    *,
    propensity_learner: Callable[[], Learner],
    outcome_learner: Callable[[], Learner],
    folds: int,
    seed: int,
    gate: IdentificationGate,
    layout: CovariateLayout | None = None,
    frozen_predictions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted AIPW scores and the overlap-retention mask.

    ``layout`` names the categorical columns of ``X`` (level codes); each
    nuisance fit then learns its level indicators on its own training
    rows, and a held-out row carrying a level absent from those rows is
    scored as the reference level and disclosed by the coded
    ``estimation.targeting.unseen_level_advisory`` warning.
    ``frozen_predictions`` supplies training-only propensity,
    treated-outcome, and control-outcome predictions, bypassing all fitting.
    Clustered honest holdout entry points set it automatically; otherwise
    ``cluster_ids`` keeps rows atomic within the cross-fitting folds.
    """
    if frozen_predictions is not None:
        e, m1, m0 = frozen_predictions
    else:
        make_propensity, make_outcome = _dr_factories(propensity_learner, outcome_learner, layout)
        labels = fold_assignments(
            unit_ids, n_folds=folds, seed=seed, stratify=d, cluster_ids=cluster_ids
        )
        e = np.empty(y.shape, dtype=float)
        m1 = np.empty(y.shape, dtype=float)
        m0 = np.empty(y.shape, dtype=float)
        unseen = UnseenLevels(y.size)
        for label in range(folds):
            val = labels == label
            train = ~val
            X_val = X[val]
            propensity = make_propensity()
            propensity.fit(X[train], d[train])
            e[val] = propensity.predict(X_val)
            unseen.record(propensity, X_val, val)

            treated = train & (d == 1.0)
            control = train & (d == 0.0)
            treated_outcome = make_outcome()
            control_outcome = make_outcome()
            treated_outcome.fit(X[treated], y[treated])
            control_outcome.fit(X[control], y[control])
            m1[val] = treated_outcome.predict(X_val)
            m0[val] = control_outcome.predict(X_val)
            unseen.record(treated_outcome, X_val, val)
            unseen.record(control_outcome, X_val, val)
        _disclose_unseen_levels(unseen)

    g = gate.min_propensity
    outside = (e < g) | (e > 1.0 - g)
    n_outside = int(outside.sum())
    if n_outside and gate.overlap == "refuse":
        _raise(
            "estimation.targeting.overlap.refuse",
            n_outside=n_outside,
            n_total=int(y.size),
            min_propensity=g,
        )
    kept = ~outside if gate.overlap == "trim" else np.ones(y.shape, dtype=bool)
    if not np.any(d[kept] == 1.0) or not np.any(d[kept] == 0.0):
        _raise(
            "estimation.targeting.overlap.empty_arm",
            n_kept=int(kept.sum()),
            n_total=int(y.size),
            min_propensity=g,
        )
    psi = np.zeros(y.shape, dtype=float)
    psi[kept] = (
        m1[kept]
        - m0[kept]
        + d[kept] * (y[kept] - m1[kept]) / e[kept]
        - (1.0 - d[kept]) * (y[kept] - m0[kept]) / (1.0 - e[kept])
    )
    return psi, kept


def _curve_weights(u: np.ndarray) -> np.ndarray:
    """Per-rank weights turning an area-under-the-curve into a linear statistic.

    For units sorted by score descending, ``TOC(j/m) = mean(psi_(1..j))
    - psibar`` and the area is ``(1/m) sum_j u_j TOC(j/m)``; exchanging
    the sums collapses this to ``sum_k w_k psi_(k)`` with
    ``w_k = (1/m)(sum_{j>=k} u_j/j - mean(u))``. ``u=1`` gives AUTOC's
    ``w_k = (H_{k,m}-1)/m``; ``u_j=j/m`` gives Qini's
    ``w_k = ((m+1)/2 - k)/m^2``. Weights sum to zero in exact
    arithmetic, so a no-signal score integrates to no area; the final
    re-centering removes float drift from the harmonic tail.
    """
    m = u.size
    j = np.arange(1, m + 1, dtype=float)
    tail = np.cumsum((u / j)[::-1])[::-1]
    w = (tail - float(u.mean())) / m
    return w - w.mean()


def _unit_weights(score: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Rank weights scattered back onto units, averaged across tied scores.

    Tied scores share the mean of ``w_k`` over the ranks they span;
    otherwise which tied unit lands in which rank (and the whole
    statistic) depends on arrival order, and a categorical-only
    interaction spec ties almost everything.
    """
    order = np.argsort(-score, kind="stable")
    ranked = _curve_weights(u)
    sorted_score = score[order]
    # Blocks of equal score, in rank order, so the averaging is summed in an
    # order that does not depend on the caller's row order either.
    starts = np.empty(score.size, dtype=bool)
    starts[0] = True
    np.not_equal(sorted_score[1:], sorted_score[:-1], out=starts[1:])
    block = np.cumsum(starts) - 1
    midweights = np.bincount(block, weights=ranked) / np.bincount(block)
    out = np.empty(score.size)
    out[order] = midweights[block]
    return out


def _rank_test(score: np.ndarray, psi: np.ndarray, u: np.ndarray) -> RankTest:
    """A rank-weighted average effect, its plug-in standard error and p-value.

    ``phi_i = m w_i psi_i`` is a per-unit influence contribution whose
    mean is the statistic, so the usual mean-of-scores SE applies. The
    weights are fixed given a score from a model that never saw these
    outcomes, which licenses treating them as constants.
    """
    m = score.size
    phi = m * _unit_weights(score, u) * psi
    estimate = float(phi.mean())
    se = float(phi.std(ddof=1)) / math.sqrt(m) if m > 1 else math.nan
    if not se > 0.0:
        # A degenerate spread carries no evidence against the null.
        return RankTest(estimate=estimate, se=se, p_value=1.0)
    return RankTest(estimate=estimate, se=se, p_value=float(norm.sf(estimate / se)))


def _group_bins(score: np.ndarray, n_groups: int) -> np.ndarray:
    """Quantile bin per unit, 0 the lowest predicted effect.

    Cuts on score values, not ranks, so tied units share a group rather
    than splitting across the cut depending on row order; a heavily
    tied score can leave groups empty, which the table reports.
    """
    edges = np.quantile(score, np.arange(1, n_groups) / n_groups)
    return np.searchsorted(edges, score, side="left")


def _sorted_groups(
    score: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    psi: np.ndarray,
    n_groups: int,
    z_crit: float,
    *,
    arm_summary: Literal["welch", "score"],
    weights: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
) -> tuple[GroupEffect, ...]:
    """Treatment-effect summary inside each predicted-effect group.

    *z_crit* is the caller's choice of critical value; ``_validation``
    Bonferroni-corrects it across ``n_groups`` so a reader scanning every
    group for the one excluding zero pays the true familywise error, not
    the per-row nominal alpha (CDDF 2018).
    """
    if cluster_ids is not None:
        assert weights is not None
        bins = _weighted_bins(score, weights, n_groups)
        groups = []
        for g in range(n_groups):
            inside = bins == g
            summary = _cluster_effect(
                y[inside], d[inside], psi[inside], weights[inside], cluster_ids[inside], arm_summary
            )
            groups.append(
                GroupEffect(
                    group=g + 1,
                    n=int(inside.sum()),
                    mean_score=_weighted_mean(score[inside], weights[inside]),
                    effect=summary.value,
                    se=summary.se,
                    lb=None,
                    ub=None,
                    n_clusters=summary.n_clusters,
                    reference_df=summary.df,
                    uncertainty_method=summary.method,
                    unavailable_reason=summary.reason,
                )
            )
        return tuple(groups)
    bins = _group_bins(score, n_groups)
    groups = []
    for g in range(n_groups):
        inside = bins == g
        if arm_summary == "welch":
            effect, se = _welch(y[inside & (d == 1.0)], y[inside & (d == 0.0)])
        else:
            effect = _mean(psi[inside])
            n_inside = int(inside.sum())
            se = float(psi[inside].std(ddof=1)) / math.sqrt(n_inside) if n_inside > 1 else math.nan
        half = z_crit * se
        groups.append(
            GroupEffect(
                group=g + 1,
                n=int(inside.sum()),
                mean_score=_mean(score[inside]),
                effect=effect,
                se=se,
                lb=effect - half,
                ub=effect + half,
            )
        )
    return tuple(groups)


def _clan_row(
    name: str,
    x: np.ndarray,
    most: np.ndarray,
    least: np.ndarray,
    z: float,
    *,
    weights: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
) -> ClanRow:
    if cluster_ids is not None:
        assert weights is not None
        summary = _cluster_contrast(
            x[most], x[least], weights[most], weights[least], cluster_ids[most], cluster_ids[least]
        )
        return ClanRow(
            covariate=name,
            mean_most=_weighted_mean(x[most], weights[most]),
            mean_least=_weighted_mean(x[least], weights[least]),
            diff=summary.value,
            se=summary.se,
            lb=None,
            ub=None,
            n_clusters=summary.n_clusters,
            reference_df=summary.df,
            uncertainty_method=summary.method,
            unavailable_reason=summary.reason,
        )
    diff, se = _welch(x[most], x[least])
    half = z * se
    return ClanRow(
        covariate=name,
        mean_most=_mean(x[most]),
        mean_least=_mean(x[least]),
        diff=diff,
        se=se,
        lb=diff - half,
        ub=diff + half,
    )


def _clan_plan(
    cols: Mapping[str, np.ndarray], covariates: Sequence[Covariate]
) -> tuple[tuple[str, np.ndarray], ...]:
    """One ``(name, value column)`` pair per CLAN row, before any interval.

    A categorical covariate contributes one row per level observed in
    the holdout, including its modal level, since this is a
    description of who is in each group, not a design matrix.
    """
    rows: list[tuple[str, np.ndarray]] = []
    for cov in covariates:
        values = cols[cov.name]
        if cov.kind == "continuous":
            rows.append((cov.name, values.astype(float)))
            continue
        as_str = values.astype(str)
        for level in np.unique(as_str):
            rows.append((f"{cov.name}={level}", (as_str == level).astype(float)))
    return tuple(rows)


def _clan(
    cols: Mapping[str, np.ndarray],
    covariates: Sequence[Covariate],
    most: np.ndarray,
    least: np.ndarray,
    z_crit: float,
    *,
    weights: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
) -> tuple[ClanRow, ...]:
    """Covariate profile of the most- versus least-affected group, at *z_crit*.

    *z_crit* is the caller's choice of critical value; ``_validation``
    Bonferroni-corrects it across ``_clan_plan``'s row count so a reader
    scanning every row for the one excluding zero pays the true
    familywise error, not the per-row nominal alpha (CDDF 2018).
    """
    return tuple(
        _clan_row(name, values, most, least, z_crit, weights=weights, cluster_ids=cluster_ids)
        for name, values in _clan_plan(cols, covariates)
    )


class _Holdout(NamedTuple):
    """The half of the units the fit never saw, scored by the half it did."""

    n_train: int
    score: np.ndarray
    y: np.ndarray
    d: np.ndarray
    cols: dict[str, np.ndarray]
    covariates: tuple[Covariate, ...]
    unit_ids: np.ndarray
    cluster_ids: np.ndarray | None = None
    psi_fn: PsiFn = _ipw_psi_fn
    cluster_weight: Literal["member_count", "equal"] = "member_count"
    bootstrap_population: _ClusterPopulation | None = None
    score_state: CateScoreState | None = None
    deployment_source_ids: np.ndarray | None = None
    #: Observational adjustment columns of the holdout rows -- numeric
    #: values and categorical level codes -- built once over every unit so
    #: training and holdout share one code table.
    adjustment: np.ndarray | None = None
    #: The level basis a caller-supplied score reads, fitted on the
    #: workflow's training rows alone; None for the built-in scores.
    basis: FittedEncoding | None = None


def _prepare_adjustment_columns(
    cols: Mapping[str, np.ndarray], adjustment: Sequence[str], *, caller: str
) -> dict[str, np.ndarray]:
    """Type every observational adjustment column: numeric values become
    floats, string values stay strings (a categorical level set), and any
    other dtype -- dates, times, nested or mixed objects -- refuses by name.
    Nulls refuse as on every CATE column: this path supports
    missing='refuse' only."""
    normalized = dict(cols)
    for name in adjustment:
        values = _column(cols, name)
        if values.dtype.kind == "O" and any(is_null(value) for value in values):
            _column({name: np.array([None], dtype=object)}, name)
        if values.dtype.kind in "biuf":
            kind: str | None = "numeric"
        elif values.dtype.kind in "US":
            kind = "categorical"
        elif values.dtype.kind == "O":
            kind = classify_objects(values)
        else:
            kind = None
        if kind is None:
            _raise(
                "estimation.targeting.adjustment_dtype",
                caller=caller,
                column=name,
                dtype=str(values.dtype),
            )
        if kind == "numeric":
            try:
                numeric = np.asarray(values, dtype=float)
            except (TypeError, ValueError, OverflowError):
                _raise(
                    "estimation.targeting.adjustment_dtype",
                    caller=caller,
                    column=name,
                    dtype=str(values.dtype),
                )
            normalized[name] = _column({name: numeric}, name)
        else:
            normalized[name] = values.astype(str)
    return normalized


def _adjustment_matrix(
    columns: Mapping[str, np.ndarray],
    adjustment: Sequence[str],
    n: int,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, CovariateLayout]:
    """The nuisance matrix over the prepared *adjustment* columns and its
    layout, coded over every unit so callers keeping different *rows* of it
    share one code table."""
    X, layout = matrix_from_columns(columns, adjustment, n)
    return (X if rows is None else X[rows]), layout


def _bind_layout(psi_fn: PsiFn, layout: CovariateLayout) -> PsiFn:
    """The built-in doubly robust score bound to *layout*, so each of its
    nuisance fits encodes the levels on its own training rows; any other
    score is returned as supplied."""
    if not (isinstance(psi_fn, functools.partial) and psi_fn.func is _dr_psi):
        return psi_fn
    return cast(PsiFn, functools.partial(_dr_psi, **{**psi_fn.keywords, "layout": layout}))


def _score_basis(
    psi_fn: PsiFn,
    layout: CovariateLayout,
    X: np.ndarray,
    rows: np.ndarray | slice = slice(None),
) -> FittedEncoding | None:
    """Fit the custom score's shared basis on training rows only.
    Built-in DR instead fits a separate basis inside each nuisance fit."""
    if psi_fn is _ipw_psi_fn or (isinstance(psi_fn, functools.partial) and psi_fn.func is _dr_psi):
        return None
    return FittedEncoding.fit(layout, X[rows])


def _score_input(basis: FittedEncoding | None, X: np.ndarray) -> np.ndarray:
    """Transform custom-score inputs and disclose unseen levels without trimming."""
    if basis is None:
        return X
    if basis.layout.categorical:
        unseen = UnseenLevels(X.shape[0])
        unseen.record_encoding(basis, X)
        _disclose_unseen_levels(unseen, score="caller-supplied score")
    return ScoreDesign(basis.transform(X), basis.names(), basis.sources())


def _guarded_design(
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    unit_ids: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate],
    adjustment: Sequence[str] = (),
    n_groups: int,
    alpha: float,
    caller: str = "_guarded_design",
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    tuple[Covariate, ...],
    np.ndarray | None,
]:
    """Coerce and validate the raw design once; every guard raises ValueError."""
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    unit_ids = np.asarray(unit_ids)
    if y.ndim != 1:
        _raise("estimation.targeting.outcome_shape", y=y.shape)
    if d.shape != y.shape:
        _raise("estimation.targeting.treatment_shape_expected", d=d.shape, y=y.shape)
    if unit_ids.shape != y.shape:
        _raise("estimation.targeting.unit_ids_shape", unit_ids=unit_ids.shape, y=y.shape)
    if unit_ids.dtype.kind not in ("U", "S", "O"):
        _raise("estimation.targeting.unit_ids_string", unit_ids=unit_ids.dtype)
    unit_ids = canonical_id_strings(unit_ids, what="unit_ids")
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape != y.shape:
            _raise(
                "estimation.targeting.cluster_ids_shape", cluster_ids=cluster_ids.shape, y=y.shape
            )
        if cluster_ids.dtype.kind not in ("U", "S", "O"):
            _raise("estimation.targeting.cluster_ids_string", cluster_ids=cluster_ids.dtype)
        cluster_ids = canonical_id_strings(cluster_ids, what="cluster_ids")
    unique_ids, id_counts = np.unique(unit_ids, return_counts=True)
    if id_counts.size and id_counts.max() > 1:
        dupes = unique_ids[id_counts > 1]
        _raise(
            "estimation.targeting.unit_ids_contains",
            count=int((id_counts > 1).sum()),
            dupes=dupes[:5].tolist(),
        )
    if not np.isfinite(y).all():
        refuse(CATE_OUTCOME_NULLS_NON)
    if not (np.isfinite(d).all() and np.isin(d, (0.0, 1.0)).all()):
        refuse(
            CATE_TREATMENT_BINARY,
            values=sorted(np.unique(d[~np.isin(d, (0.0, 1.0))]).tolist())[:5],
        )
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    if n_groups < 2:
        _raise("estimation.targeting.n_groups_least", n_groups=n_groups)
    covariates = _dedup((*interact, *adjust))
    columns = {}
    for cov in covariates:
        values = _column(cols, cov.name)
        if values.size != y.size:
            _raise(
                "estimation.targeting.covariate_rows_expected",
                cov=cov.name,
                values=values.size,
                y=y.size,
            )
        columns[cov.name] = values
    for name in adjustment:
        if name in columns:
            continue
        values = _column(cols, name)
        if values.size != y.size:
            _raise(
                "estimation.targeting.covariate_rows_expected",
                cov=name,
                values=values.size,
                y=y.size,
            )
        columns[name] = values
    columns = _prepare_adjustment_columns(columns, adjustment, caller=caller)
    return y, d, columns, unit_ids, covariates, cluster_ids


def _fitted_scores(fit: CateResult, cols: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Score known rows, including an intercept-only fit with no feature columns."""
    state = fit.score_state
    return state.score(cols) if state.basis.transforms else np.full(n, state.intercept)


# Identity metadata stays separate from model inputs throughout the honest split.
def _honest_holdout(  # noqa: PLR0913
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    unit_ids: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    psi_fn: PsiFn = _ipw_psi_fn,
    intervention_grain: Literal["unit", "cluster"] = "unit",
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate],
    adjustment: Sequence[str] = (),
    n_groups: int,
    alpha: float,
    holdout_mask: np.ndarray | None = None,
    training_mask: np.ndarray | None = None,
    basis: FittedEncoding | None = None,
    caller: str = "_honest_holdout",
) -> _Holdout:
    """Guard the inputs, split the units, fit on the train side, score the rest.

    The split is the crc32 content hash by default; *holdout_mask* injects a
    caller-chosen partition instead (the nested-selection outer test).
    *training_mask* can narrow the complementary training side while leaving
    the holdout untouched, with the same per-arm guards either way. When
    *cluster_ids* is supplied, both the generated and any caller-supplied
    *holdout_mask* are enforced cluster-atomic - a cluster split across
    train/holdout is refused with a stable coded error naming it - before
    the model is ever fit. Randomized training eligibility must also retain
    complete clusters; observational overlap trimming retains its unit target.
    A caller-supplied score reads its categorical levels through one basis
    fitted on training rows alone: *basis* when the caller fitted it already
    (the nested selection fits it on the inner half, this split's training
    side, before any overlap trim), otherwise fitted here on the train side.
    """
    _validate_cluster_weight(cluster_weight, clustered=cluster_ids is not None)
    y, d, columns, unit_ids, covariates, cluster_ids = _guarded_design(
        y,
        d,
        cols,
        unit_ids,
        cluster_ids=cluster_ids,
        interact=interact,
        adjust=adjust,
        adjustment=adjustment,
        n_groups=n_groups,
        alpha=alpha,
        caller=caller,
    )
    if holdout_mask is None:
        hold = _holdout_mask(unit_ids, cluster_ids=cluster_ids)
    else:
        hold = np.asarray(holdout_mask, dtype=bool)
        if hold.shape != y.shape:
            _raise("estimation.targeting.holdout_mask_shape", hold=hold.shape, y=y.shape)
    if cluster_ids is not None:
        check_cluster_atomic(cluster_ids, hold)
    train = ~hold
    if training_mask is not None:
        eligible = np.asarray(training_mask, dtype=bool)
        if eligible.shape != y.shape:
            _raise("estimation.targeting.training_mask_shape", train=eligible.shape, y=y.shape)
        train &= eligible
        if cluster_ids is not None and psi_fn is _ipw_psi_fn:
            check_cluster_atomic(cluster_ids, train)
    for name, half in (("training", train), ("holdout", hold)):
        n_treated = int((d[half] == 1.0).sum())
        n_control = int(half.sum()) - n_treated
        if n_treated < 2 or n_control < 2:
            _raise(
                "estimation.targeting.half_split_holds",
                name=name,
                n_treated=n_treated,
                n_control=n_control,
            )

    fit = fit_cate(
        y[train],
        d[train],
        {name: values[train] for name, values in columns.items()},
        interact=interact,
        adjust=adjust,
        alpha=alpha,
        cluster_ids=None if cluster_ids is None else cluster_ids[train],
        intervention_grain=intervention_grain,
        cluster_weight=cluster_weight,
    )
    hold_cols = {name: values[hold] for name, values in columns.items()}
    X, layout = _adjustment_matrix(columns, adjustment, y.size)
    psi_fn = _bind_layout(psi_fn, layout)
    if basis is None:
        basis = _score_basis(psi_fn, layout, X, train)
    else:
        # One code table serves the caller's basis and these holdout rows.
        assert basis.layout == layout
    if cluster_ids is not None:
        psi_fn = _freeze_psi_fn(psi_fn, y[train], d[train], X[train], X[hold])
    return _Holdout(
        n_train=int(train.sum()),
        score=_fitted_scores(fit, hold_cols, int(hold.sum())),
        score_state=fit.score_state,
        y=y[hold],
        d=d[hold],
        cols=hold_cols,
        covariates=covariates,
        unit_ids=unit_ids[hold],
        cluster_ids=None if cluster_ids is None else cluster_ids[hold],
        psi_fn=psi_fn,
        cluster_weight=cluster_weight,
        adjustment=X[hold],
        basis=basis,
    )


def _holdout_adjustment_matrix(holdout: _Holdout) -> np.ndarray:
    """The adjustment matrix of the holdout rows as `holdout.psi_fn` reads it."""
    assert holdout.adjustment is not None
    return _score_input(holdout.basis, holdout.adjustment)


def _check_deployment_overlap(
    kept: np.ndarray, ids: np.ndarray | None, grain: Literal["unit", "cluster"]
) -> None:
    """Refuse partial rosters before pooling scores or evaluating cluster budgets."""
    if grain == "cluster" and ids is not None and not kept.all():
        if np.intersect1d(ids[kept], ids[~kept]).size:
            _raise("estimation.targeting.overlap.partial_cluster")


def _trim_holdout(holdout: _Holdout, kept: np.ndarray) -> _Holdout:
    """Apply an overlap mask to every aligned holdout value."""
    psi_fn = holdout.psi_fn
    if isinstance(psi_fn, functools.partial) and psi_fn.func is _dr_psi:
        predictions = psi_fn.keywords.get("frozen_predictions")
        if predictions is not None:
            trimmed = tuple(values[kept] for values in predictions)
            for values in trimmed:
                values.setflags(write=False)
            psi_fn = cast(PsiFn, functools.partial(psi_fn, frozen_predictions=trimmed))
    return _Holdout(
        n_train=holdout.n_train,
        score=holdout.score[kept],
        score_state=holdout.score_state,
        y=holdout.y[kept],
        d=holdout.d[kept],
        cols={name: values[kept] for name, values in holdout.cols.items()},
        covariates=holdout.covariates,
        unit_ids=holdout.unit_ids[kept],
        cluster_ids=None if holdout.cluster_ids is None else holdout.cluster_ids[kept],
        psi_fn=psi_fn,
        cluster_weight=holdout.cluster_weight,
        adjustment=None if holdout.adjustment is None else holdout.adjustment[kept],
        basis=holdout.basis,
    )


def _validation(
    holdout: _Holdout,
    psi: np.ndarray,
    *,
    n_groups: int,
    alpha: float,
    arm_summary: Literal["welch", "score"] = "welch",
    split_caveat: str = _SINGLE_SPLIT_CAVEAT,
    population: str | None = None,
    evaluation_population: CateEvaluationPopulation | None = None,
    bootstrap_seed: int = 0,
    bootstrap_repetitions: int = 999,
) -> CateValidation:
    """Every reported number, read off the held-out half and nothing else."""
    if holdout.cluster_ids is not None:
        return _cluster_validation(
            holdout,
            psi,
            n_groups=n_groups,
            alpha=alpha,
            evaluation_population=evaluation_population,
            arm_summary=arm_summary,
            split_caveat=split_caveat,
            population=population,
            bootstrap_seed=bootstrap_seed,
            bootstrap_repetitions=bootstrap_repetitions,
        )
    score, y_hold, d_hold = holdout.score, holdout.y, holdout.d
    z_crit = two_sided_critical_value(norm.isf, alpha, what="targeting validation holdout ATE")
    if arm_summary == "welch":
        ate, ate_se = _welch(y_hold[d_hold == 1.0], y_hold[d_hold == 0.0])
    else:
        ate = _mean(psi)
        ate_se = float(psi.std(ddof=1)) / math.sqrt(psi.size) if psi.size > 1 else math.nan
    # GATES is a family of n_groups simultaneous intervals; Bonferroni-correct
    # across it rather than cut every one at the unadjusted per-row alpha.
    group_z = two_sided_critical_value(
        norm.isf, alpha / n_groups, what="targeting validation GATES"
    )
    groups = _sorted_groups(score, y_hold, d_hold, psi, n_groups, group_z, arm_summary=arm_summary)

    m = score.size
    autoc = _rank_test(score, psi, np.ones(m))
    qini = _rank_test(score, psi, np.arange(1, m + 1, dtype=float) / m)

    bins = _group_bins(score, n_groups)
    # Same correction for CLAN, across the row count it actually reports
    # (one per continuous covariate, one per observed categorical level).
    clan_plan = _clan_plan(holdout.cols, holdout.covariates)
    clan_z = (
        two_sided_critical_value(norm.isf, alpha / len(clan_plan), what="targeting validation CLAN")
        if clan_plan
        else z_crit
    )
    clan = _clan(holdout.cols, holdout.covariates, bins == n_groups - 1, bins == 0, clan_z)

    n_clusters = None if holdout.cluster_ids is None else int(np.unique(holdout.cluster_ids).size)
    return CateValidation(
        evaluation_population=evaluation_population,
        n_train=holdout.n_train,
        n_holdout=m,
        holdout_ate=Estimate(
            value=ate, lb=ate - z_crit * ate_se, ub=ate + z_crit * ate_se, level=1.0 - alpha
        ),
        groups=groups,
        autoc=autoc,
        qini=qini,
        clan=clan,
        alpha=alpha,
        passed=autoc.p_value is not None and autoc.p_value < alpha,
        population=population,
        split_caveat=split_caveat,
        n_clusters=n_clusters,
    )


def validate_cate_arrays(  # noqa: PLR0913
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    unit_ids: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap_seed: int = 0,
    bootstrap_repetitions: int = 999,
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate] = (),
    n_groups: int = 5,
    alpha: float = 0.05,
    adjustment: Sequence[str] = (),
    psi_fn: PsiFn = _ipw_psi_fn,
    arm_summary: Literal["welch", "score"] = "welch",
    intervention_grain: Literal["unit", "cluster"] = "unit",
    include_evaluation_population: bool = False,
) -> CateValidation:
    """Fit a CATE model on half the units and validate it on the other half.

    The split is a content hash of ``unit_ids`` or, when supplied,
    ``cluster_ids``. Every row in a cluster lands on the same side,
    with the cluster's assignment broadcast to all its members. Half A fits
    :func:`~increment.estimation.cate.fit_cate`, half B is scored by the
    fit and supplies every reported number - reusing fitting rows for
    any of it would inflate the top group and rank tests together.

    CLAN profiles every covariate named in *interact* or *adjust* (all
    pre-exposure by contract). Observational *adjustment* columns are numeric
    or categorical (string values, one level per distinct string); a null
    refuses. The built-in doubly robust score fits each categorical's
    modal-reference level indicators inside every nuisance fit.
    ``psi_fn`` must accept five positional arguments
    ``(y, d, X, unit_ids, cluster_ids)`` aligned to the holdout rows, where
    ``X`` holds the *adjustment* columns and ``cluster_ids`` is ``None``
    when omitted. A caller-supplied score receives ``X`` as a
    :class:`ScoreDesign`: each categorical as the 0/1 indicators of a basis
    fitted on the training half alone (the modal training level is the
    reference), ``X.columns`` naming every column; a holdout level absent
    from the training half reads as the reference and is disclosed by the
    coded ``estimation.targeting.unseen_level_advisory`` warning, never
    trimmed. It returns ``(psi, kept_mask)``: one score and one boolean
    retention flag per holdout row. Every guard raises ``ValueError``
    naming the problem; none returns a partial result.

    With clusters, ``cluster_weight`` chooses unit or equal-cluster mass.
    ``bootstrap_seed`` and ``bootstrap_repetitions`` control heldout-only
    resampling and are recorded with availability and cluster counts. Built-in
    DR nuisance predictions are frozen from training-only fits; custom
    five-argument score callables must supply their own frozen nuisances.
    """
    _validate_cluster_weight(cluster_weight, clustered=cluster_ids is not None)
    _validate_bootstrap(bootstrap_seed, bootstrap_repetitions)

    holdout = _honest_holdout(
        y,
        d,
        cols,
        unit_ids,
        cluster_ids=cluster_ids,
        cluster_weight=cluster_weight,
        psi_fn=psi_fn,
        intervention_grain=intervention_grain,
        interact=interact,
        adjust=adjust,
        adjustment=adjustment,
        n_groups=n_groups,
        alpha=alpha,
        caller="validate_cate_arrays",
    )
    psi, kept = _holdout_scores(holdout)
    population = None
    if not kept.all():
        holdout = _trim_holdout(holdout, kept)
        psi = psi[kept]
        population = "overlap_subpopulation"
    evaluation_population = (
        _evaluation_population(
            holdout,
            split="honest",
            seed=None,
            retention="overlap_trimmed" if population else "all",
            overlap=population,
        )
        if include_evaluation_population
        else None
    )
    holdout = _prepare_cluster_bootstrap(holdout, arm_summary)
    return _validation(
        holdout,
        psi,
        n_groups=n_groups,
        alpha=alpha,
        arm_summary=arm_summary,
        evaluation_population=evaluation_population,
        population=population,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )


def _toc_at(score: np.ndarray, psi: np.ndarray, j: int) -> RankTest:
    """The targeting curve at the top *j* units: ``mean(psi of top j) - psibar``.

    A spike of height ``m`` at rank *j* in :func:`_curve_weights`'s
    ``u`` collapses the area integral to one point: ``w_k = 1/j - 1/m``
    on the top *j*, ``-1/m`` below - exactly the TOC, with the same
    influence-function SE the rank tests use.
    """
    u = np.zeros(score.size)
    u[j - 1] = float(score.size)
    return _rank_test(score, psi, u)


def _required_columns(
    interact: Sequence[Covariate], adjust: Sequence[Covariate], adjustment: Sequence[str]
) -> tuple[str, ...]:
    """Covariate columns a caller must supply to reproduce a policy's score:
    `interact` + `adjust` (declaration order, de-duplicated by name), plus
    any observational `adjustment` column not already named. Cluster
    identity is never a member -- it rides alongside a score as metadata,
    never a column the score itself reads.
    """
    names = [c.name for c in _dedup((*interact, *adjust))]
    return tuple(dict.fromkeys((*names, *adjustment)))


def _deployment(
    score: np.ndarray,
    fraction: float,
    ids: np.ndarray | None,
    weighting: Literal["member_count", "equal"],
    grain: Literal["unit", "cluster"],
    *,
    source_ids: np.ndarray | None = None,
) -> Deployment:
    if grain == "cluster":
        assert ids is not None
        return cluster_prefix(score, ids, fraction, weighting, source_ids=source_ids)
    weights = np.ones(score.size) if ids is None else _target_weights(ids, weighting)
    if fraction == 0 or not score.size:
        return Deployment(np.zeros(score.size, dtype=bool), None, 0.0)
    threshold = (
        float(np.quantile(score, 1 - fraction))
        if ids is None
        else _weighted_cut(score, weights, 1 - fraction)
    )
    targeted = score >= threshold
    return Deployment(targeted, threshold, float(weights[targeted].sum() / weights.sum()))


class _PolicyMetadata(TypedDict):
    intervention_grain: Literal["unit", "cluster"]
    deploy_grain: Literal["unit", "cluster"]
    cluster_weight: Literal["member_count", "equal"]
    budget_rule: Literal["unit_threshold", "whole_cluster_prefix"]
    achieved_fraction: float
    score_cutoff: float | None
    score_state: CateScoreState
    required_columns: tuple[str, ...]


def _policy_metadata(
    holdout: _Holdout,
    deployment: Deployment,
    intervention_grain: Literal["unit", "cluster"],
    deploy_grain: Literal["unit", "cluster"],
    required_columns: tuple[str, ...],
) -> _PolicyMetadata:
    if holdout.score_state is None:
        _raise("estimation.targeting.score_state_required")
    return _PolicyMetadata(
        intervention_grain=intervention_grain,
        deploy_grain=deploy_grain,
        cluster_weight=holdout.cluster_weight,
        budget_rule="whole_cluster_prefix" if deploy_grain == "cluster" else "unit_threshold",
        achieved_fraction=deployment.achieved_fraction,
        score_cutoff=deployment.threshold,
        score_state=holdout.score_state,
        required_columns=required_columns,
    )


def _rule_at_fraction(
    holdout: _Holdout,
    validation: CateValidation,
    psi: np.ndarray,
    *,
    fraction: float,
    alpha: float,
    arm_summary: Literal["welch", "score"],
    intervention_grain: Literal["unit", "cluster"] = "unit",
    deploy_grain: Literal["unit", "cluster"] | None = None,
    required_columns: tuple[str, ...] = (),
) -> TargetingRule:
    """Evaluate exactly the deployment candidate, conditional on the honest gate."""
    grain = resolve_deploy_grain(
        intervention_grain, deploy_grain, clustered=holdout.cluster_ids is not None
    )
    if holdout.cluster_ids is not None:
        return _cluster_rule(
            holdout,
            validation,
            psi,
            fraction=fraction,
            alpha=alpha,
            arm_summary=arm_summary,
            intervention_grain=intervention_grain,
            deploy_grain=grain,
            required_columns=required_columns,
        )
    deployment = _deployment(holdout.score, fraction, None, holdout.cluster_weight, grain)
    metadata = _policy_metadata(holdout, deployment, intervention_grain, grain, required_columns)
    if not validation.passed or not deployment.targeted.any():
        return TargetingRule(
            fraction=fraction,
            recommendation="target" if validation.passed else "simple",
            validation=validation,
            threshold=None,
            policy_value=None,
            uplift_vs_average=None,
            unavailable_reason=_reason("empty_group") if not deployment.targeted.any() else None,
            population=validation.population,
            n_clusters=validation.n_clusters,
            **metadata,
        )
    targeted, threshold, _ = deployment
    y_hold, d_hold = holdout.y, holdout.d
    if arm_summary == "welch":
        n_treatment = int((targeted & (d_hold == 1.0)).sum())
        n_control = int((targeted & (d_hold == 0.0)).sum())
        if n_treatment < 2 or n_control < 2:
            _raise(
                "estimation.targeting.targeting_fraction_leaves",
                fraction=fraction,
                n_treatment=n_treatment,
                n_control=n_control,
            )
        value, _ = _welch(y_hold[targeted & (d_hold == 1.0)], y_hold[targeted & (d_hold == 0.0)])
    else:
        value = _mean(psi[targeted])
    toc = _toc_at(holdout.score, psi, int(targeted.sum()))
    assert toc.estimate is not None
    return TargetingRule(
        fraction=fraction,
        recommendation="target",
        validation=validation,
        threshold=threshold,
        policy_value=Estimate(value=value),
        uplift_vs_average=Estimate(value=toc.estimate),
        population=validation.population,
        n_clusters=validation.n_clusters,
        **metadata,
    )


# Public array API exposes model, identity and policy inputs independently.
def targeting_rule_arrays(  # noqa: PLR0913
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    unit_ids: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap_seed: int = 0,
    bootstrap_repetitions: int = 999,
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate] = (),
    fraction: float,
    n_groups: int = 5,
    alpha: float = 0.05,
    adjustment: Sequence[str] = (),
    psi_fn: PsiFn = _ipw_psi_fn,
    arm_summary: Literal["welch", "score"] = "welch",
    intervention_grain: Literal["unit", "cluster"] = "unit",
    deploy_grain: Literal["unit", "cluster"] | None = None,
    include_evaluation_population: bool = False,
) -> TargetingRule:
    """The deployable answer: treat the top *fraction*, or treat everyone alike.

    Runs :func:`validate_cate_arrays`'s honest split, then - only if
    the gate passed - cuts the held-out score at its ``1 - fraction``
    quantile and reports what the rule delivered on units the fit
    never saw. A failed gate returns a ``"simple"`` recommendation with
    the validation attached and no policy numbers.

    Observational *adjustment* columns are numeric or categorical, exactly as
    for :func:`validate_cate_arrays`.

    *fraction* has no default: choosing the cut after seeing the group
    table costs 60-120% upward bias, so it is a pre-commitment. The
    holdout outcomes decide whether the gate opens and supply the
    policy value behind it, so conditional on a gate that only passed
    by chance the policy value runs high; unconditionally it is
    unbiased. The gate protects against a fabricated ranking, not the
    selection its own threshold induces -- the reported
    ``policy_value``/``uplift_vs_average`` therefore ship as a point
    value only (no interval): a nominal z-interval built at *alpha*
    would print as if unconditional, which it is not, this being a
    single non-confirmatory look, not a repeatable calibrated interval.

    ``deploy_grain`` defaults only from ``intervention_grain``. Cluster
    deployment pools member scores and uses a whole-cluster prefix budget;
    unit deployment keeps individual scores even with dependence IDs.
    Fractions zero/one select nobody/everybody. Cluster weights determine
    both the budget and evaluation target, including every bootstrap draw.

    With clusters, ``cluster_weight`` chooses unit or equal-cluster mass.
    ``bootstrap_seed`` and ``bootstrap_repetitions`` control heldout-only
    resampling and are recorded with availability and cluster counts. Built-in
    DR nuisance predictions are frozen from training-only fits; custom
    five-argument score callables must supply their own frozen nuisances.
    """
    deploy_grain = resolve_deploy_grain(
        intervention_grain, deploy_grain, clustered=cluster_ids is not None
    )
    _validate_cluster_weight(cluster_weight, clustered=cluster_ids is not None)
    _validate_bootstrap(bootstrap_seed, bootstrap_repetitions)

    if not 0.0 <= fraction <= 1.0:
        _raise("estimation.targeting.fraction_share_units", fraction=fraction)
    holdout = _honest_holdout(
        y,
        d,
        cols,
        unit_ids,
        cluster_ids=cluster_ids,
        cluster_weight=cluster_weight,
        psi_fn=psi_fn,
        intervention_grain=intervention_grain,
        interact=interact,
        adjust=adjust,
        adjustment=adjustment,
        n_groups=n_groups,
        alpha=alpha,
        caller="targeting_rule_arrays",
    )
    psi, kept = _holdout_scores(holdout)
    _check_deployment_overlap(kept, holdout.cluster_ids, deploy_grain)
    population = None
    if not kept.all():
        holdout = _trim_holdout(holdout, kept)
        psi = psi[kept]
        population = "overlap_subpopulation"
    evaluation_population = (
        _evaluation_population(
            holdout,
            split="honest",
            seed=None,
            retention="overlap_trimmed" if population else "all",
            overlap=population,
        )
        if include_evaluation_population
        else None
    )
    holdout = _prepare_cluster_bootstrap(holdout, arm_summary)
    validation = _validation(
        holdout,
        psi,
        n_groups=n_groups,
        alpha=alpha,
        arm_summary=arm_summary,
        population=population,
        evaluation_population=evaluation_population,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    return _rule_at_fraction(
        holdout,
        validation,
        psi,
        fraction=fraction,
        alpha=alpha,
        arm_summary=arm_summary,
        intervention_grain=intervention_grain,
        deploy_grain=deploy_grain,
        required_columns=_required_columns(interact, adjust, adjustment),
    )


def _selection_nuisance_scores(
    psi_fn: PsiFn,
    y_in: np.ndarray,
    d_in: np.ndarray,
    X_in: np.ndarray,
    unit_ids_in: np.ndarray,
    cluster_ids_in: np.ndarray | None,
    *,
    layout: CovariateLayout,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, FittedEncoding | None]:
    """Selection-half psi scores, the overlap mask, any fold labels used, and
    the level basis a caller-supplied score read.

    Built-in DR fits frozen nuisances on each fold's training complement and
    scores that fold's held-out rows, matching the outer honest split's
    frozen-nuisance contract; ``layout`` names the level-code columns of
    ``X_in`` those fits encode on their own training rows. A custom
    five-argument callable keeps its existing single-call contract; fold
    assignment is left to the caller. Its basis is fitted on the whole inner
    half -- the outer test's training side -- and returned so the outer
    test hands the callable the same columns.
    """
    if not (isinstance(psi_fn, functools.partial) and psi_fn.func is _dr_psi):
        basis = _score_basis(psi_fn, layout, X_in)
        psi_in, kept = cast(_ArrayPsiFn, psi_fn)(
            y_in, d_in, _score_input(basis, X_in), unit_ids_in, cluster_ids_in
        )
        return psi_in, kept, None, basis
    psi_fn = _bind_layout(psi_fn, layout)
    folds = fold_assignments(
        unit_ids_in, n_folds=n_folds, seed=seed, stratify=d_in, cluster_ids=cluster_ids_in
    )
    psi_in = np.empty(y_in.size)
    kept = np.ones(y_in.size, dtype=bool)
    for label in range(n_folds):
        val = folds == label
        frozen = _freeze_psi_fn(psi_fn, y_in[~val], d_in[~val], X_in[~val], X_in[val])
        held_out_ids = None if cluster_ids_in is None else cluster_ids_in[val]
        psi_in[val], kept[val] = cast(_ArrayPsiFn, frozen)(
            y_in[val], d_in[val], X_in[val], unit_ids_in[val], held_out_ids
        )
    return psi_in, kept, folds, None


def _trim_selection_overlap(
    kept: np.ndarray,
    inner: np.ndarray,
    y_in: np.ndarray,
    d_in: np.ndarray,
    cols_in: dict[str, np.ndarray],
    unit_ids_in: np.ndarray,
    cluster_ids_in: np.ndarray | None,
    X_in: np.ndarray,
    psi_in: np.ndarray,
    folds: np.ndarray | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    str | None,
]:
    """Drop selection-half rows the overlap gate rejected, keeping fold labels aligned.

    Returns the trimmed selection-half arrays plus the outer-holdout
    ``training_mask`` (aligned to every unit, so :func:`_honest_holdout` can
    intersect it with its own overlap trim) and the resulting population
    label. Both are ``None`` and every array passes through unchanged when
    nothing was trimmed.
    """
    if kept.all():
        return y_in, d_in, cols_in, unit_ids_in, cluster_ids_in, X_in, psi_in, folds, None, None
    training_mask = inner.copy()
    training_mask[inner] = kept
    trimmed_cols = {name: values[kept] for name, values in cols_in.items()}
    trimmed_cluster_ids = None if cluster_ids_in is None else cluster_ids_in[kept]
    trimmed_folds = None if folds is None else folds[kept]
    return (
        y_in[kept],
        d_in[kept],
        trimmed_cols,
        unit_ids_in[kept],
        trimmed_cluster_ids,
        X_in[kept],
        psi_in[kept],
        trimmed_folds,
        training_mask,
        "overlap_subpopulation",
    )


def _unclustered_selection_table(
    fracs: tuple[float, ...],
    folds: np.ndarray,
    cols_in: Mapping[str, np.ndarray],
    psi_in: np.ndarray,
    cost_per_treated: float,
    fit: Callable[[np.ndarray], CateResult],
    *,
    alpha: float,
) -> tuple[int, tuple[FractionScore, ...]]:
    """Out-of-fold net-benefit table and its argmax fraction, unclustered path.

    The argmax of several noisy per-fraction estimates carries its own
    winner's-curse-inflated interval at any nominal level, so it ships as a
    point value; the outer half's ``rule`` is the reportable verdict for the
    chosen fraction.
    """

    targeted_counts = np.zeros(len(fracs), dtype=int)

    def _evaluate(model: CateResult, val: np.ndarray, index: int) -> np.ndarray:
        score = _fitted_scores(
            model, {name: values[val] for name, values in cols_in.items()}, int(val.sum())
        )
        targeted = _deployment(score, fracs[index], None, "member_count", "unit").targeted
        targeted_counts[index] += int(targeted.sum())
        return np.where(targeted, psi_in[val] - cost_per_treated, 0.0)

    selection = select_out_of_fold(len(fracs), folds, fit, _evaluate)
    best_index = selection.best_index
    z_crit = two_sided_critical_value(norm.isf, alpha, what="targeting selection net benefit")

    def _net_benefit(index: int, est: float, se: float) -> Estimate:
        if len(fracs) > 1 and index == best_index:
            return Estimate(value=est)
        return Estimate(value=est, lb=est - z_crit * se, ub=est + z_crit * se, level=1.0 - alpha)

    inner_table = tuple(
        FractionScore(
            fraction=f,
            achieved_fraction=float(targeted_counts[i] / n),
            net_benefit=_net_benefit(i, est, se),
            n=n,
        )
        for i, (f, est, se, n) in enumerate(
            zip(fracs, selection.estimates, selection.ses, selection.n_units, strict=True)
        )
    )
    return best_index, inner_table


def select_targeting_rule_arrays(  # noqa: PLR0913
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    unit_ids: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    bootstrap_seed: int = 0,
    bootstrap_repetitions: int = 999,
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate] = (),
    fractions: Sequence[float],
    cost_per_treated: float = 0.0,
    n_folds: int = 5,
    seed: int,
    n_groups: int = 5,
    alpha: float = 0.05,
    adjustment: Sequence[str] = (),
    psi_fn: PsiFn = _ipw_psi_fn,
    arm_summary: Literal["welch", "score"] = "welch",
    intervention_grain: Literal["unit", "cluster"] = "unit",
    deploy_grain: Literal["unit", "cluster"] | None = None,
    include_evaluation_population: bool = False,
) -> TargetingSelection:
    """Choose a fraction on inner folds, then score the locked rule once outside.

    A seeded, arm-stratified half becomes the outer test, unread until
    the fraction is locked. On the inner half, each of *n_folds* models
    is fit without its validation fold and scores it; each grid
    fraction's net benefit ``E[1{targeted}(tau - cost_per_treated)]`` is
    a plain mean of the identification design's per-unit score contributions.
    The argmax (ties to the smaller fraction) is locked, the model refit on
    the same retained inner population, and the outer half supplies the same
    gate and policy numbers :func:`targeting_rule_arrays` reports.

    Observational *adjustment* columns are numeric or categorical, exactly as
    for :func:`validate_cate_arrays`.

    *fractions* and *seed* have no defaults: the grid and split are
    pre-commitments, like ``targeting_rule``'s fraction.

    ``deploy_grain`` defaults only from ``intervention_grain``. Cluster
    deployment pools member scores and uses a whole-cluster prefix budget;
    unit deployment keeps individual scores even with dependence IDs.
    Fractions zero/one select nobody/everybody. Cluster weights determine
    both the budget and evaluation target, including every bootstrap draw.

    With clusters, ``cluster_weight`` chooses unit or equal-cluster mass.
    ``bootstrap_seed`` and ``bootstrap_repetitions`` control heldout-only
    resampling and are recorded with availability and cluster counts. Built-in
    DR nuisance predictions are frozen from training-only fits; custom
    five-argument score callables must supply their own frozen nuisances,
    and read their categorical levels through one :class:`ScoreDesign` basis
    fitted on the inner half, in the inner call and the outer test alike.
    """
    deploy_grain = resolve_deploy_grain(
        intervention_grain, deploy_grain, clustered=cluster_ids is not None
    )
    _validate_cluster_weight(cluster_weight, clustered=cluster_ids is not None)
    _validate_bootstrap(bootstrap_seed, bootstrap_repetitions)

    fracs = tuple(sorted(float(f) for f in fractions))
    if not fracs:
        _raise("estimation.targeting.fractions_non_empty")
    for f in fracs:
        if not 0.0 <= f <= 1.0:
            _raise("estimation.targeting.every_fraction", f=f)
    if len(set(fracs)) != len(fracs):
        _raise("estimation.targeting.fractions_contains_duplicates", fracs=fracs)
    if n_folds < 2:
        refuse(N_FOLDS_LEAST, n_folds=n_folds)

    y, d, columns, unit_ids, _, cluster_ids = _guarded_design(
        y,
        d,
        cols,
        unit_ids,
        cluster_ids=cluster_ids,
        interact=interact,
        adjust=adjust,
        adjustment=adjustment,
        n_groups=n_groups,
        alpha=alpha,
        caller="select_targeting_rule_arrays",
    )
    outer = outer_split(unit_ids, test_size=0.5, seed=seed, stratify=d, cluster_ids=cluster_ids)
    inner = ~outer
    y_in, d_in = y[inner], d[inner]
    cols_in = {name: values[inner] for name, values in columns.items()}
    unit_ids_in = unit_ids[inner]
    cluster_ids_in = None if cluster_ids is None else cluster_ids[inner]
    # Coded over every unit, so the inner half's basis and the outer test's
    # rows agree on every level code.
    X_in, layout = _adjustment_matrix(columns, adjustment, y.size, rows=inner)
    psi_in, kept, folds, basis = _selection_nuisance_scores(
        psi_fn,
        y_in,
        d_in,
        X_in,
        unit_ids_in,
        cluster_ids_in,
        layout=layout,
        n_folds=n_folds,
        seed=seed,
    )
    _check_deployment_overlap(kept, cluster_ids_in, deploy_grain)
    (
        y_in,
        d_in,
        cols_in,
        unit_ids_in,
        cluster_ids_in,
        X_in,
        psi_in,
        folds,
        training_mask,
        selection_population,
    ) = _trim_selection_overlap(
        kept,
        inner,
        y_in,
        d_in,
        cols_in,
        unit_ids_in,
        cluster_ids_in,
        X_in,
        psi_in,
        folds,
    )

    if folds is None:
        folds = fold_assignments(
            unit_ids_in, n_folds=n_folds, seed=seed, stratify=d_in, cluster_ids=cluster_ids_in
        )
    for label in range(n_folds):
        val = folds == label
        n_treated = int((d_in[val] == 1.0).sum())
        n_control = int(val.sum()) - n_treated
        if n_treated < 2 or n_control < 2:
            _raise(
                "estimation.targeting.validation_fold_holds",
                label=label,
                n_treated=n_treated,
                n_control=n_control,
            )

    if psi_fn is _ipw_psi_fn:
        # Preserve the randomized path's fold-local centering exactly while
        # keeping candidate evaluation on the same precomputed psi[val] seam.
        for label in range(n_folds):
            val = folds == label
            psi_in[val], _ = _ipw_psi_fn(
                y_in[val],
                d_in[val],
                X_in[val],
                unit_ids_in[val],
                None if cluster_ids_in is None else cluster_ids_in[val],
            )

    def _fit(train: np.ndarray) -> CateResult:
        return fit_cate(
            y_in[train],
            d_in[train],
            {name: values[train] for name, values in cols_in.items()},
            interact=interact,
            adjust=adjust,
            alpha=alpha,
            cluster_ids=None if cluster_ids_in is None else cluster_ids_in[train],
            intervention_grain=intervention_grain,
            cluster_weight=cluster_weight,
        )

    inner_score = np.empty(y_in.size)
    if cluster_ids_in is not None:
        # Freeze each out-of-fold score before choosing a candidate or bootstrapping.
        for label in range(n_folds):
            val = folds == label
            model = _fit(~val)
            inner_score[val] = model.score_state._score_for_roster(
                {name: values[val] for name, values in cols_in.items()}, cluster_ids_in[val]
            )[0]
        estimates = _cluster_net_benefits(
            inner_score,
            psi_in,
            y_in,
            d_in,
            cluster_ids_in,
            folds,
            fracs,
            cost_per_treated,
            cluster_weight,
            deploy_grain=deploy_grain,
            randomized_score=psi_fn is _ipw_psi_fn,
        )
        selection_values = [estimate.value for estimate in estimates]
        if any(value is None or not math.isfinite(value) for value in selection_values):
            _raise("estimation.targeting.nonfinite_statistic")
        best_index = int(np.argmax([cast("float", value) for value in selection_values]))
    else:
        best_index, inner_table = _unclustered_selection_table(
            fracs,
            folds,
            cols_in,
            psi_in,
            cost_per_treated,
            _fit,
            alpha=alpha,
        )

    holdout = _honest_holdout(
        y,
        d,
        columns,
        unit_ids,
        cluster_ids=cluster_ids,
        cluster_weight=cluster_weight,
        psi_fn=psi_fn,
        intervention_grain=intervention_grain,
        interact=interact,
        adjust=adjust,
        adjustment=adjustment,
        n_groups=n_groups,
        alpha=alpha,
        holdout_mask=outer,
        training_mask=training_mask,
        basis=basis,
        caller="select_targeting_rule_arrays",
    )
    psi, kept = _holdout_scores(holdout)
    _check_deployment_overlap(kept, holdout.cluster_ids, deploy_grain)
    # All CATE/nuisance fits, including the outer training fit, precede any bootstrap.
    if cluster_ids_in is not None:
        inner_table = _cluster_inner_table(
            inner_score,
            psi_in,
            y_in,
            d_in,
            cluster_ids_in,
            folds,
            fracs,
            cost_per_treated,
            cluster_weight,
            deploy_grain=deploy_grain,
            randomized_score=psi_fn is _ipw_psi_fn,
            best_index=best_index,
            alpha=alpha,
            seed=bootstrap_seed,
            repetitions=bootstrap_repetitions,
            arm_summary=arm_summary,
        )
    population = None
    if not kept.all():
        holdout = _trim_holdout(holdout, kept)
        psi = psi[kept]
        population = "overlap_subpopulation"
    evaluation_population = (
        _evaluation_population(
            holdout,
            split="outer",
            seed=int(seed),
            retention="overlap_trimmed" if population else "all",
            overlap=population,
        )
        if include_evaluation_population
        else None
    )
    holdout = _prepare_cluster_bootstrap(holdout, arm_summary)
    validation = _validation(
        holdout,
        psi,
        n_groups=n_groups,
        alpha=alpha,
        arm_summary=arm_summary,
        split_caveat=_SEEDED_STRATIFIED_SPLIT_CAVEAT,
        evaluation_population=evaluation_population,
        population=population,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    return TargetingSelection(
        fractions=fracs,
        inner=inner_table,
        selected_fraction=fracs[best_index],
        cost_per_treated=float(cost_per_treated),
        n_folds=n_folds,
        seed=int(seed),
        rule=_rule_at_fraction(
            holdout,
            validation,
            psi,
            fraction=fracs[best_index],
            alpha=alpha,
            arm_summary=arm_summary,
            intervention_grain=intervention_grain,
            deploy_grain=deploy_grain,
            required_columns=_required_columns(interact, adjust, adjustment),
        ),
        population=selection_population,
        n_clusters=None if cluster_ids_in is None else int(np.unique(cluster_ids_in).size),
        cluster_weight=cluster_weight if cluster_ids_in is not None else None,
        uncertainty_method="bootstrap-t+cluster-jackknife-t"
        if cluster_ids_in is not None
        else None,
        covariance_method=(
            inner_table[best_index].covariance_method if cluster_ids_in is not None else None
        ),
        reference_df=(inner_table[best_index].reference_df if cluster_ids_in is not None else None),
        support_failures=inner_table[best_index].support_failures
        if cluster_ids_in is not None
        else (),
        bootstrap_valid_repetitions=(
            inner_table[best_index].bootstrap_valid_repetitions
            if cluster_ids_in is not None
            else None
        ),
        unavailable_reason=(
            inner_table[best_index].unavailable_reason if cluster_ids_in is not None else None
        ),
        bootstrap_seed=bootstrap_seed if cluster_ids_in is not None else None,
        bootstrap_repetitions=bootstrap_repetitions if cluster_ids_in is not None else None,
    )


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
    "estimation.targeting.unseen_level_advisory",
    IncrementWarning,
    lambda *, unseen, n, score="doubly robust score": (
        f"{score} unseen-level advisory: {unseen_levels_text(unseen, n)}"
    ),
)


def _disclose_unseen_levels(unseen: UnseenLevels, *, score: str | None = None) -> None:
    """Warn once per fit stage about the scored rows whose level the fit
    that scored them -- or, named by ``score``, the caller-supplied score's
    training-fitted basis -- never saw; nothing is gated or trimmed."""
    summary = unseen.summary()
    if summary:
        if score is None:
            _warn("estimation.targeting.unseen_level_advisory", unseen=summary, n=unseen.n)
        else:
            _warn(
                "estimation.targeting.unseen_level_advisory",
                unseen=summary,
                n=unseen.n,
                score=score,
            )


# Clustered validation deliberately leaves the legacy row-statistic branch intact.
_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.targeting.cluster_count_bounds": "n_clusters ({n_clusters}) must not exceed n_holdout ({n_holdout})",
        "estimation.targeting.cluster_count_mismatch": "rule n_clusters ({n_clusters}) must equal validation.n_clusters ({validation_n_clusters})",
        "estimation.targeting.cluster_grain_count_required": "cluster-grain policies require a non-null n_clusters",
        "estimation.targeting.unit_ids_string": "unit_ids must be string-typed for a stable split, got dtype {unit_ids!r} -- cast to str before calling (e.g. unit_frame's own unit_id column, which is already String-typed)",
        "estimation.targeting.cluster_ids_string": "cluster_ids must be string-typed for a stable split, got dtype {cluster_ids!r} -- cast to str before calling",
        "estimation.targeting.outcome_shape": "outcome y must be 1-d, got shape {y}",
        "estimation.targeting.treatment_shape_expected": "treatment d has shape {d}, expected {y} to match y",
        "estimation.targeting.unit_ids_shape": "unit_ids has shape {unit_ids}, expected {y} to match y",
        "estimation.targeting.unit_ids_contains": "unit_ids contains {count} duplicate id(s), e.g. {dupes!r} -- a duplicated unit can land on both sides of the honest split (or twice on one side), breaking the independence the split relies on; deduplicate before calling",
        "estimation.targeting.n_groups_least": "n_groups must be at least 2 to compare a top group against a bottom one, got {n_groups}",
        "estimation.targeting.covariate_rows_expected": "covariate {cov!r} has {values} rows, expected {y}",
        "estimation.targeting.adjustment_dtype": "{caller} cannot adjust on observational adjustment column {column!r} of dtype {dtype}: an adjustment column is numeric (int/float/bool) or categorical (string); a date, time, nested or mixed-type column carries no adjustment meaning. Derive a numeric or categorical pre-exposure column upstream",
        "estimation.targeting.holdout_mask_shape": "holdout_mask has shape {hold}, expected {y} to match y",
        "estimation.targeting.training_mask_shape": "training_mask has shape {train}, expected {y} to match y",
        "estimation.targeting.cluster_ids_shape": "cluster_ids has shape {cluster_ids}, expected {y} to match y",
        "estimation.targeting.half_split_holds": "the {name} half of the split holds {n_treated} treated and {n_control} control unit(s); an honest split needs at least 2 per arm in each half. Estimate the average effect instead of validating a CATE model",
        "estimation.targeting.targeting_fraction_leaves": "targeting fraction={fraction!r} leaves fewer than 2 held-out units in an arm (treatment={n_treatment}, control={n_control}); increase the pre-committed fraction or supply more validation units",
        "estimation.targeting.fraction_share_units": "fraction must be in [0, 1] -- the share of units the rule would treat, got {fraction}. It has no default on purpose: picking the cut after seeing the group table biases the reported policy value upward by 60-120%",
        "estimation.targeting.fractions_non_empty": "fractions must be a non-empty pre-specified grid -- selecting from a grid chosen after seeing results is the bias this function exists to prevent",
        "estimation.targeting.every_fraction": "every fraction must be in [0, 1], got {f}",
        "estimation.targeting.fractions_contains_duplicates": "fractions contains duplicates: {fracs}",
        "estimation.targeting.validation_fold_holds": "validation fold {label} holds {n_treated} treated and {n_control} control unit(s); selection needs at least 2 per arm in every fold -- use fewer folds or estimate the average effect instead",
        "estimation.targeting.overlap.refuse": RefusalSpec(
            "estimation.targeting.overlap.refuse",
            IdentificationError,
            lambda *, n_outside, n_total, min_propensity: (
                f"{n_outside} of {n_total} out-of-fold propensities fall outside "
                f"[{min_propensity}, {1.0 - min_propensity}]; the overlap gate refused estimation"
            ),
        ),
        "estimation.targeting.overlap.empty_arm": RefusalSpec(
            "estimation.targeting.overlap.empty_arm",
            IdentificationError,
            template="overlap trimming at min_propensity={min_propensity} kept {n_kept} of {n_total} units but emptied a treatment arm",
        ),
        "estimation.targeting.policy_metadata": "Policy target, grain, budget rule and fitted cutoff must agree",
        "estimation.targeting.policy_grain_mismatch": "Prediction cannot change the fitted policy deployment grain",
        "estimation.targeting.score_state_required": "A deployable policy requires its frozen fitted scoring state",
        "estimation.targeting.evaluation_population_invalid": "Invalid evaluation population: {reason}",
        "estimation.targeting.overlap.partial_cluster": RefusalSpec(
            "estimation.targeting.overlap.partial_cluster",
            IdentificationError,
            template="Cluster deployment cannot trim only some members of a cluster for overlap",
        ),
        "estimation.targeting.bootstrap_options": "bootstrap_seed must be a nonnegative integer and bootstrap_repetitions an integer >= 2",
    },
)

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
_raise = raiser(_REFUSALS)
for _code, _message in (
    ("empty_group", "The requested group has no positive target mass"),
    ("insufficient_clusters", "At least two independent clusters are required in each mean"),
    ("degenerate_cluster_variance", "Centered cluster contributions have zero variance"),
    ("nonfinite_statistic", "The statistic or its covariance is not finite"),
    ("degenerate_rank_distribution", "All held-out predicted scores are tied"),
    (
        "bootstrap_insufficient_clusters",
        "A bootstrap group has no source clusters",
    ),
    ("empty_arm", "A required treatment arm has no source clusters"),
    ("insufficient_arm_clusters", "A required treatment arm has fewer than two source clusters"),
    ("bootstrap_empty_arm", "A bootstrap draw empties a required treatment arm"),
    (
        "bootstrap_unavailable_replicate",
        "At least one bootstrap replicate has unavailable uncertainty",
    ),
    ("bootstrap_zero_variance", "The centered bootstrap distribution has zero variance"),
    ("bootstrap_tail_resolution", "The requested tail is smaller than the bootstrap resolution"),
    ("jackknife_unavailable_replicate", "A delete-cluster statistic is unavailable"),
):
    _REFUSALS[f"estimation.targeting.{_code}"] = RefusalSpec(
        f"estimation.targeting.{_code}", InvalidRequestError, lambda message=_message: message
    )


def _reason(name: str) -> str:
    return _REFUSALS[f"estimation.targeting.{name}"].code


def _validate_bootstrap(seed: object, repetitions: object) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, (int, np.integer))
        or repetitions < 2
    ):
        _raise("estimation.targeting.bootstrap_options")


class ClusterBootstrap(CodedModel, BaseModel):
    """Whole-cluster resampling controls shared by every honest-validation entry point.

    Bundles the seed and repetition count of the heldout-only whole-cluster
    bootstrap that :func:`validate_cate_arrays`, :func:`targeting_rule_arrays`
    and :func:`select_targeting_rule_arrays` use to recompute ranks, empirical
    GATES/CLAN cutoffs and policy values on every replicate. Immutable:
    construct once and reuse across calls. ``seed`` and ``repetitions`` that
    are not a genuine nonnegative integer and an integer ``>= 2`` -- a bool, a
    float, or a forged/mutated instance -- refuse with the coded
    ``estimation.targeting.bootstrap_options`` error before any nuisance
    model is fit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    seed: int = 0
    repetitions: int = 999

    @model_validator(mode="before")
    @classmethod
    def _reject_forged_options(cls, data: object) -> object:
        if isinstance(data, Mapping):
            _validate_bootstrap(data.get("seed", 0), data.get("repetitions", 999))
        return data


@overload
def _freeze_psi_fn(
    psi_fn: _ArrayPsiFn, y: np.ndarray, d: np.ndarray, X: np.ndarray, X_hold: np.ndarray
) -> _ArrayPsiFn: ...


@overload
def _freeze_psi_fn(
    psi_fn: PsiFn, y: np.ndarray, d: np.ndarray, X: np.ndarray, X_hold: np.ndarray
) -> PsiFn: ...


def _freeze_psi_fn(
    psi_fn: PsiFn, y: np.ndarray, d: np.ndarray, X: np.ndarray, X_hold: np.ndarray
) -> PsiFn:
    """Fit built-in DR nuisances once on training rows, before holdout evaluation.

    A custom five-argument callable remains supported and is evaluated once;
    it is responsible for supplying scores from externally frozen nuisances.
    The returned built-in partial accepts exactly the original five arguments.
    Holdout rows carrying a level absent from the training rows are scored
    as the reference level and disclosed by the coded advisory.
    """
    if not isinstance(psi_fn, functools.partial) or psi_fn.func is not _dr_psi:
        return psi_fn
    kwargs = dict(psi_fn.keywords)
    make_propensity, make_outcome = _dr_factories(
        kwargs["propensity_learner"], kwargs["outcome_learner"], kwargs.get("layout")
    )
    propensity = make_propensity()
    propensity.fit(X, d)
    treated = make_outcome()
    control = make_outcome()
    treated.fit(X[d == 1], y[d == 1])
    control.fit(X[d == 0], y[d == 0])
    predictions = tuple(
        np.array(model.predict(X_hold), dtype=float, copy=True)
        for model in (propensity, treated, control)
    )
    unseen = UnseenLevels(X_hold.shape[0])
    for model in (propensity, treated, control):
        unseen.record(model, X_hold)
    _disclose_unseen_levels(unseen)
    for values in predictions:
        values.setflags(write=False)
    kwargs["frozen_predictions"] = predictions
    return cast(PsiFn, functools.partial(_dr_psi, **kwargs))


def _holdout_scores(holdout: _Holdout) -> tuple[np.ndarray, np.ndarray]:
    if holdout.cluster_ids is not None and holdout.psi_fn is _ipw_psi_fn:
        weights = _target_weights(holdout.cluster_ids, holdout.cluster_weight)
        return _ipw_psi(holdout.y, holdout.d, weights=weights), np.ones(holdout.y.size, dtype=bool)
    return cast(_ArrayPsiFn, holdout.psi_fn)(
        holdout.y,
        holdout.d,
        _holdout_adjustment_matrix(holdout),
        holdout.unit_ids,
        holdout.cluster_ids,
    )


def _target_weights(ids: np.ndarray, weighting: Literal["member_count", "equal"]) -> np.ndarray:
    _, labels, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return np.ones(ids.size) if weighting == "member_count" else 1.0 / counts[labels]


def _roundoff_gamma(n: int) -> float:
    # Covers normalization, centered subtraction, products and two accumulations.
    rounding = (8 * n + 16) * np.finfo(float).eps
    return float(rounding / (1 - rounding))


def _exact_cluster_contributions(
    y: np.ndarray, weights: np.ndarray, inverse: np.ndarray, k: int
) -> tuple[Fraction, list[Fraction]]:
    """Resolve cancellation using the exact values of the supplied binary64 inputs."""
    anchor = Fraction(float(y[0]))
    if not np.any(y != y[0]) and np.isfinite(weights).all() and np.any(weights != 0):
        # Every centered term is exactly zero, so the rational sums below are too.
        return anchor, [Fraction(0)] * k
    mass = [Fraction(0) for _ in range(k)]
    totals = [Fraction(0) for _ in range(k)]
    for value, weight, group in zip(y, weights, inverse, strict=True):
        w = Fraction(float(weight))
        mass[group] += w
        totals[group] += w * (Fraction(float(value)) - anchor)
    total_mass = sum(mass, Fraction(0))
    delta = sum(totals, Fraction(0)) / total_mass
    return anchor + delta, [
        (total - m * delta) / total_mass for total, m in zip(totals, mass, strict=True)
    ]


class _CenteredMean(NamedTuple):
    """A weighted mean about the first value, with the arrays that produced it."""

    mean: float
    centered: np.ndarray
    p: np.ndarray
    delta: float
    spread: np.ndarray


def _cluster_contributions(
    moments: _CenteredMean, inverse: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    centered, p = moments.centered, moments.p
    u = np.bincount(inverse, weights=p * (centered - moments.delta), minlength=k)
    magnitude = p * moments.spread
    scale = np.bincount(inverse, weights=magnitude, minlength=k)
    mass = np.bincount(inverse, weights=p, minlength=k)
    # Absolute centered products bound local sums and propagated mean error.
    bound = _roundoff_gamma(centered.size) * (scale + mass * magnitude.sum())
    return u, bound


def _centered_mean(values: np.ndarray, weights: np.ndarray) -> _CenteredMean | None:
    if not values.size or not np.isfinite(values).all() or not np.sum(weights) > 0:
        return None
    # Center before summation to retain variation about a large common offset.
    centered = values - values[0]
    p = weights / weights.sum()
    delta = float(np.dot(p, centered))
    mean = float(values[0] + delta)
    spread = np.abs(centered)
    bound = _roundoff_gamma(values.size) * float(np.dot(p, spread))
    if math.isfinite(mean) and min(abs(mean), abs(delta)) <= bound:
        exact, _ = _exact_cluster_contributions(
            values, weights, np.zeros(values.size, dtype=int), 1
        )
        mean = float(exact)
    return _CenteredMean(mean, centered, p, delta, spread) if math.isfinite(mean) else None


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    moments = _centered_mean(values, weights)
    return None if moments is None else moments.mean


class _ClusterSummary(NamedTuple):
    value: float | None
    se: float | None
    df: float | None
    n_clusters: int
    method: str
    reason: str | None = None


def _contribution_se(u: np.ndarray, k: int, copies: np.ndarray) -> float:
    # u[g] totals copies[g] identical instances, each contributing u[g] / copies[g].
    scale = float(np.max(np.abs(u)))
    if scale == 0 or not math.isfinite(scale):
        return scale
    normalized = u / scale
    squares = float(np.dot(normalized / np.maximum(copies, 1), normalized))
    return scale * math.sqrt(squares * k / (k - 1))


def _cluster_codes(ids: np.ndarray, copies: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Dense cluster codes and instance copies per code; ids are codes when copies is given.

    A code with copies c stands for c identical clusters, as when a resample
    draws one source cluster c times; its rows carry c times the target mass.
    """
    if copies is not None:
        return ids, copies
    labels, inverse = np.unique(ids, return_inverse=True)
    return inverse, np.ones(labels.size)


def _instance_count(codes: np.ndarray, copies: np.ndarray) -> int:
    return int(copies[np.bincount(codes, minlength=copies.size) > 0].sum())


def _cluster_mean(
    y: np.ndarray,
    weights: np.ndarray,
    ids: np.ndarray,
    *,
    method: str = "pooled-DR-influence",
    copies: np.ndarray | None = None,
) -> _ClusterSummary:
    inverse, copies = _cluster_codes(ids, copies)
    k = _instance_count(inverse, copies)
    moments = _centered_mean(y, weights)
    if moments is None:
        return _ClusterSummary(
            None,
            None,
            None,
            k,
            method,
            _reason("empty_group" if not y.size else "nonfinite_statistic"),
        )
    mean = moments.mean
    if k < 2:
        return _ClusterSummary(mean, None, None, k, method, _reason("insufficient_clusters"))
    u, bound = _cluster_contributions(moments, inverse, copies.size)
    if np.all(np.abs(u) <= bound):
        exact_mean, exact_u = _exact_cluster_contributions(y, weights, inverse, copies.size)
        mean, u = float(exact_mean), np.array([float(v) for v in exact_u])
    se = _contribution_se(u, k, copies)
    return _cluster_uncertainty(mean, se, float(k - 1), k, method)


def _cluster_uncertainty(
    value: float, se: float, df: float, k: int, method: str
) -> _ClusterSummary:
    if not math.isfinite(value) or not math.isfinite(se):
        return _ClusterSummary(None, None, None, k, method, _reason("nonfinite_statistic"))
    if se <= 0:
        return _ClusterSummary(value, None, None, k, method, _reason("degenerate_cluster_variance"))
    return _ClusterSummary(value, se, df, k, method)


def _cluster_contrast(
    a: np.ndarray,
    b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    *,
    copies: np.ndarray | None = None,
) -> _ClusterSummary:
    """Ratio-mean contrast: disjoint-arm Welch or joint signed cluster scores.

    For shared clusters, sum u_a,g - u_b,g BEFORE squaring and use K/(K-1)
    over the union. For disjoint sets, use each arm's own K and Welch df.
    Weights are inherited from the full target population, never redefined
    using subgroup cluster sizes.
    """
    if copies is None:
        labels = np.union1d(ids_a, ids_b)
        ia, ib = np.searchsorted(labels, ids_a), np.searchsorted(labels, ids_b)
        copies = np.ones(labels.size)
    else:
        ia, ib = ids_a, ids_b
    in_a = np.bincount(ia, minlength=copies.size) > 0
    in_b = np.bincount(ib, minlength=copies.size) > 0
    k = int(copies[in_a | in_b].sum())
    overlap = bool(np.any(in_a & in_b))
    method = "overlap-signed-combined" if overlap else "disjoint-arm-Welch"
    ma, mb = _centered_mean(a, weights_a), _centered_mean(b, weights_b)
    if ma is None or mb is None:
        return _ClusterSummary(
            None,
            None,
            None,
            k,
            method,
            _reason("empty_group" if not a.size or not b.size else "nonfinite_statistic"),
        )
    try:
        difference = math.fsum((float(a[0]), -float(b[0]), ma.delta, -mb.delta))
    except OverflowError:
        return _ClusterSummary(None, None, None, k, method, _reason("nonfinite_statistic"))
    ka, kb = int(copies[in_a].sum()), int(copies[in_b].sum())
    if ka < 2 or kb < 2:
        return _ClusterSummary(difference, None, None, k, method, _reason("insufficient_clusters"))
    ua, ba = _cluster_contributions(ma, ia, copies.size)
    ub, bb = _cluster_contributions(mb, ib, copies.size)
    combined = ua - ub
    bound = ba + bb + np.finfo(float).eps * (np.abs(ua) + np.abs(ub))
    if np.all(np.abs(combined) <= bound):
        same = (
            np.array_equal(a, b) and np.array_equal(weights_a, weights_b) and np.array_equal(ia, ib)
        )
        if same:
            # One population on both sides: its exact contrast is identically zero.
            difference, combined = 0.0, np.zeros(copies.size)
        else:
            ea, eua = _exact_cluster_contributions(a, weights_a, ia, copies.size)
            eb, eub = _exact_cluster_contributions(b, weights_b, ib, copies.size)
            difference = float(ea - eb)
            ua, ub = np.array([float(v) for v in eua]), np.array([float(v) for v in eub])
            combined = np.array([float(x - y) for x, y in zip(eua, eub, strict=True)])
    if overlap:
        se = _contribution_se(combined, k, copies)
        df = float(k - 1)
    else:
        sa = _contribution_se(ua, ka, copies)
        sb = _contribution_se(ub, kb, copies)
        se = math.hypot(sa, sb)
        # Normalize before squaring to retain very small or large uncertainties.
        df = (
            1 / ((sa / se) ** 4 / (ka - 1) + (sb / se) ** 4 / (kb - 1))
            if se > 0 and math.isfinite(se)
            else 0.0
        )
    return _cluster_uncertainty(difference, se, float(df), k, method)


def _cluster_effect(
    y: np.ndarray,
    d: np.ndarray,
    psi: np.ndarray,
    weights: np.ndarray,
    ids: np.ndarray,
    arm_summary: Literal["welch", "score"],
    *,
    copies: np.ndarray | None = None,
) -> _ClusterSummary:
    if arm_summary == "score":
        return _cluster_mean(psi, weights, ids, copies=copies)
    a, b = d == 1, d == 0
    return _cluster_contrast(y[a], y[b], weights[a], weights[b], ids[a], ids[b], copies=copies)


def _cluster_interval(summary: _ClusterSummary, alpha: float) -> tuple[float | None, float | None]:
    if summary.value is None or summary.se is None or summary.df is None:
        return None, None
    critical = two_sided_critical_value(student_t_isf, alpha, summary.df, what="cluster validation")
    half = critical * summary.se
    lb, ub = summary.value - half, summary.value + half
    if not math.isfinite(lb) or not math.isfinite(ub):
        _raise("estimation.targeting.nonfinite_statistic")
    return lb, ub


def _summary_estimate(summary: _ClusterSummary, alpha: float) -> Estimate | None:
    if summary.value is None:
        return None
    lb, ub = _cluster_interval(summary, alpha)
    return Estimate(
        value=summary.value,
        lb=lb,
        ub=ub,
        level=1 - alpha if lb is not None else None,
        alpha=alpha if lb is not None else None,
    )


def _weighted_cut(score: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Inverse empirical weighted CDF; equal scores share a deterministic cut.

    A tied block is kept whole (as in the legacy deployment threshold). This
    convention is independent of input order and uniform within-cluster cloning.
    """
    values, inverse = np.unique(score, return_inverse=True)
    return _block_cut(values, np.bincount(inverse, weights=weights), quantile, score.size)


def _block_cut(values: np.ndarray, mass: np.ndarray, quantile: float, rows: int) -> float:
    """Cut on ascending distinct scores with positive masses summed from *rows* rows."""
    cdf = np.cumsum(mass) / mass.sum()
    # Positive sums/division have relative error bounded by gamma_(4N+8).
    # Include a boundary within that bound on either side, including row clones.
    rounding = (4 * rows + 8) * np.finfo(float).eps
    gamma = rounding / (1 - rounding)
    index = np.searchsorted(cdf * (1 + gamma), quantile, side="left")
    return float(values[min(int(index), values.size - 1)])


def _weighted_bins(score: np.ndarray, weights: np.ndarray, n_groups: int) -> np.ndarray:
    edges = [_weighted_cut(score, weights, g / n_groups) for g in range(1, n_groups)]
    return np.searchsorted(edges, score, side="left")


def _rank_components(
    score: np.ndarray, psi: np.ndarray, weights: np.ndarray
) -> tuple[float, float, tuple[np.ndarray, np.ndarray]]:
    """Integrate weighted empirical TOC, interpolating uniformly within ties.

    On a score block (a,b], TOC(q)=block_mean-overall+(R-a*block_mean)/q,
    where R is the cumulative weighted response before the block. Integrate
    TOC(q) and q*TOC(q) exactly. This distributional definition is invariant
    to uniform row cloning; all tied units receive the same rank treatment.

    For a fixed rank weight r, the response covariance functional is
    E[r psi] - E[r] E[psi]. Its centered influence is
    (r-E[r])(psi-E[psi]) - Cov(r,psi). Return the uncentered products;
    _cluster_mean centers and sums them within clusters before squaring.
    This is a conditional response reference for studentization, not an
    influence formula for estimated ranks. The bootstrap recomputes both.
    """
    _, inverse = np.unique(-score, return_inverse=True)
    curve = _block_rank_curve(inverse, psi, weights)
    return curve.autoc, curve.qini, _rank_references(curve, inverse)


class _RankCurve(NamedTuple):
    """Integrated rank values, with the block and row arrays their references reuse."""

    autoc: float
    qini: float
    mass: np.ndarray
    cumulative: np.ndarray
    a: np.ndarray
    logs: np.ndarray
    centered: np.ndarray
    p: np.ndarray


def _block_rank_curve(inverse: np.ndarray, psi: np.ndarray, weights: np.ndarray) -> _RankCurve:
    """Rank values given each row's tied block, numbered from the top score down."""
    mass = np.bincount(inverse, weights=weights)
    centered = psi - psi[0]
    response = np.bincount(inverse, weights=weights * centered) / mass
    mass = mass / mass.sum()
    overall = float(np.dot(mass, response))
    p = weights / weights.sum()
    scale = float(np.dot(p, np.abs(centered)))
    if np.all(np.abs(response - overall) <= _roundoff_gamma(psi.size) * scale):
        if mass.size == 1:
            # One tied block holds all the mass: its exact contribution is zero.
            response = np.zeros(1)
        else:
            _, exact_u = _exact_cluster_contributions(psi, weights, inverse, mass.size)
            response = np.array([float(v) for v in exact_u]) / mass
        overall = float(np.dot(mass, response))
    cumulative = np.cumsum(mass)
    a = np.concatenate(([0.0], cumulative[:-1]))
    prior = np.concatenate(([0.0], np.cumsum(mass * response)[:-1]))
    correction = prior - a * response
    logs = np.zeros(mass.size)
    logs[1:] = np.log1p(mass[1:] / a[1:])
    autoc = np.sum((response - overall) * mass + correction * logs)
    qini = np.sum((response - overall) * mass * (cumulative + a) / 2 + correction * mass)
    return _RankCurve(float(autoc), float(qini), mass, cumulative, a, logs, centered, p)


def _rank_references(curve: _RankCurve, inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row rank-response products, uncentered as `_rank_components` describes."""
    # Average -log(q)-1 and 1/2-q over each tied probability block.
    autoc_weight = -np.log(curve.cumulative) - (curve.a / curve.mass) * curve.logs
    qini_weight = (1 - curve.cumulative - curve.a) / 2
    residual = curve.centered - float(np.dot(curve.p, curve.centered))
    autoc_reference, qini_reference = (
        (rank_weight[inverse] - float(np.dot(curve.mass, rank_weight))) * residual
        for rank_weight in (autoc_weight, qini_weight)
    )
    return autoc_reference, qini_reference


def _weighted_ranks(score: np.ndarray, psi: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    autoc, qini, _ = _rank_components(score, psi, weights)
    return autoc, qini


class _ClusterPopulation(NamedTuple):
    members: tuple[np.ndarray, ...]
    strata: tuple[np.ndarray, ...]


def _cluster_population(
    ids: np.ndarray,
    d: np.ndarray,
    *,
    stratified: bool,
    folds: np.ndarray | None = None,
) -> _ClusterPopulation:
    """Group rows once, preserving sorted labels and original within-cluster order."""
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    members = tuple(np.split(order, np.cumsum(counts)[:-1]))
    pure = all(np.unique(d[rows]).size == 1 for rows in members)
    strata: dict[tuple[int, float], list[int]] = {}
    for g, rows in enumerate(members):
        key = (
            int(folds[rows[0]]) if folds is not None else 0,
            float(d[rows[0]]) if stratified and pure else 0.0,
        )
        strata.setdefault(key, []).append(g)
    return _ClusterPopulation(members, tuple(np.array(pool) for pool in strata.values()))


def _prepare_cluster_bootstrap(
    holdout: _Holdout,
    arm_summary: Literal["welch", "score"],
) -> _Holdout:
    if holdout.cluster_ids is None:
        return holdout
    return holdout._replace(
        bootstrap_population=_cluster_population(
            holdout.cluster_ids,
            holdout.d,
            stratified=arm_summary == "welch",
        )
    )


def _cluster_resample(population: _ClusterPopulation, rng: np.random.Generator) -> np.ndarray:
    """Draw source clusters with replacement, separately within each stratum.

    The prepared population preserves arm strata for pure randomized clusters,
    and fold strata for inner selection; mixed clusters and DR are unstratified.
    """
    return np.concatenate(
        [rng.choice(pool, size=len(pool), replace=True) for pool in population.strata]
    )


def _cluster_draw(
    population: _ClusterPopulation,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw whole clusters with a new instance ID for every repeated draw."""
    members = population.members
    draws = _cluster_resample(population, rng)
    rows = np.concatenate([members[g] for g in draws])
    instances = np.concatenate([np.full(members[g].size, j) for j, g in enumerate(draws)])
    return rows, instances


class _BootstrapResult(NamedTuple):
    se: float | None
    p_value: float | None
    lb: float | None
    ub: float | None
    valid: int
    reason: str | None


def _cluster_deletions(population: _ClusterPopulation) -> Iterator[np.ndarray]:
    """Delete each source cluster once, retaining all remaining member rows."""
    if len(population.members) < 2:
        return
    for omitted in range(len(population.members)):
        yield np.concatenate(
            [rows for index, rows in enumerate(population.members) if index != omitted]
        )


def _delete_cluster_summary(
    estimate: float | None,
    deleted: Sequence[float | None],
    population: _ClusterPopulation,
) -> _ClusterSummary:
    """CV3 delete-cluster scale, centered on the full estimate, not the delete mean.

    For a ratio mean, T - T_(-g) = u_g / (1 - h_g), where h_g is
    cluster mass / total mass. Complete deletions also move ranks and cuts.
    """
    k = len(population.members)
    method = "delete-cluster-CV3"
    if k < 2 or any(len(pool) < 2 for pool in population.strata):
        return _ClusterSummary(estimate, None, None, k, method, _reason("insufficient_clusters"))
    if (
        estimate is None
        or not math.isfinite(estimate)
        or len(deleted) != k
        or any(value is None or not math.isfinite(value) for value in deleted)
    ):
        return _ClusterSummary(
            estimate, None, None, k, method, _reason("jackknife_unavailable_replicate")
        )
    factor = math.sqrt((k - 1) / k)
    differences = []
    for value in deleted:
        assert value is not None
        difference = estimate - value
        differences.append(
            factor * difference if math.isfinite(difference) else factor * estimate - factor * value
        )
    se = math.hypot(*differences)
    # The smallest independent resampling stratum limits the t reference.
    df = float(min(len(pool) - 1 for pool in population.strata))
    return _cluster_uncertainty(estimate, se, df, k, method)


def _jackknife_hull(
    bootstrap: _BootstrapResult,
    summary: _ClusterSummary,
    alpha: float,
    *,
    family: int = 1,
) -> _BootstrapResult:
    """Union bootstrap-t and delete-cluster-t intervals; intersect their rejections.

    Containment preserves either component's coverage, when that component is
    valid. It does not turn either asymptotic method into a finite-K guarantee.
    The public se remains the complete bootstrap SD.
    """
    if bootstrap.reason not in {None, _reason("bootstrap_tail_resolution")}:
        return bootstrap
    if summary.reason is not None:
        return _BootstrapResult(None, None, None, None, bootstrap.valid, summary.reason)
    assert summary.value is not None and summary.se is not None and summary.df is not None
    assert bootstrap.p_value is not None
    p = max(bootstrap.p_value, float(student_t.sf(summary.value / summary.se, summary.df)))
    if bootstrap.reason is not None:
        return bootstrap._replace(p_value=p)
    assert bootstrap.lb is not None and bootstrap.ub is not None
    tail = Fraction(alpha) / (2 * family)
    rounded_tail = float(tail)
    if Fraction(rounded_tail) > tail:
        rounded_tail = math.nextafter(rounded_tail, 0.0)
    if rounded_tail <= 0:
        return _BootstrapResult(
            bootstrap.se, None, None, None, bootstrap.valid, _reason("bootstrap_tail_resolution")
        )
    half = student_t_isf(rounded_tail, summary.df) * summary.se
    lower, upper = summary.value - half, summary.value + half
    if not math.isfinite(lower) or not math.isfinite(upper):
        return _BootstrapResult(
            None, None, None, None, bootstrap.valid, _reason("nonfinite_statistic")
        )
    return bootstrap._replace(
        p_value=p,
        lb=min(bootstrap.lb, lower),
        ub=max(bootstrap.ub, upper),
    )


def _bootstrap_bound(estimate: float, scale: float, root: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        direct = estimate - scale * root
    if math.isfinite(direct):
        return direct
    exact = Fraction(estimate) - Fraction(scale) * Fraction(root)
    try:
        return float(exact)
    except OverflowError:
        return math.copysign(math.inf, 1 if exact > 0 else -1)


def _centered_bootstrap(
    estimate: float | None,
    values: Sequence[float | None],
    alpha: float,
    *,
    scale: float | None,
    scales: Sequence[float | None],
    family: int = 1,
    reason: str | None = None,
) -> _BootstrapResult:
    """Invert bootstrap-t roots (T* - T) / s*, using the original scale s.

    Center at the empirical target T, not E*(T*): the latter deletes estimated
    bias, including ratio/cutoff bias. The positive covariance references s and
    s* need not estimate the full rank variance for first-order validity: if
    sqrt(K)s and sqrt(K)s* converge to the same positive limit, Slutsky's
    theorem transfers consistency of the complete-statistic bootstrap to its
    scaled roots. This alone promises neither refinement nor finite-K coverage.

    Report the complete-statistic bootstrap SD as se, not the reference scale.
    Never discard a failed root. Exact rational tail allocation rounds order
    statistics outward and retains the (B+1) Monte Carlo correction.
    """
    valid = sum(v is not None and math.isfinite(v) for v in values)
    if estimate is not None and not math.isfinite(estimate):
        reason = _reason("nonfinite_statistic")
    if reason is None and (scale is None or not math.isfinite(scale) or scale <= 0):
        reason = _reason("degenerate_cluster_variance")
    if reason is None and (estimate is None or valid != len(values) or valid < 2):
        reason = _reason("bootstrap_unavailable_replicate")
    if reason is not None:
        return _BootstrapResult(None, None, None, None, valid, reason)
    assert estimate is not None and scale is not None
    assert len(scales) == len(values)
    samples = np.array(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        centered = samples - samples[0]
        deviations = centered - centered.mean()
        bound = _roundoff_gamma(samples.size) * (np.abs(centered) + np.abs(centered).mean())
    se = math.hypot(*deviations) / math.sqrt(samples.size - 1)
    if not math.isfinite(se) or np.all(np.abs(deviations) <= bound):
        # Resolve overflow and ambiguous spread from the exact binary64 inputs.
        exact = [Fraction(float(v)) for v in samples]
        center = sum(exact, Fraction(0)) / len(exact)
        exact_deviations = [v - center for v in exact]
        spread = max(abs(v) for v in exact_deviations)
        normalized_se = (
            math.hypot(*(float(v / spread) for v in exact_deviations)) / math.sqrt(samples.size - 1)
            if spread
            else 0.0
        )
        try:
            se = float(spread * Fraction(normalized_se))
        except OverflowError:
            return _BootstrapResult(None, None, None, None, valid, _reason("nonfinite_statistic"))
    if not math.isfinite(se):
        return _BootstrapResult(None, None, None, None, valid, _reason("nonfinite_statistic"))
    if se <= 0:
        return _BootstrapResult(None, None, None, None, valid, _reason("bootstrap_zero_variance"))
    replicate_scales = np.array(
        [s if s is not None and math.isfinite(s) and s > 0 else scale for s in scales],
        dtype=float,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        roots = (samples - estimate) / replicate_scales
    for index in np.flatnonzero(~np.isfinite(roots)):
        try:
            roots[index] = float(
                (Fraction(float(samples[index])) - Fraction(estimate))
                / Fraction(float(replicate_scales[index]))
            )
        except OverflowError:
            pass
    if not np.isfinite(roots).all():
        return _BootstrapResult(None, None, None, None, valid, _reason("nonfinite_statistic"))
    # Rank tests are one-sided; intervals are two-sided using the same null.
    probability = Fraction(1 + int(np.count_nonzero(roots >= estimate / scale)), len(values) + 1)
    p = float(probability)
    if Fraction(p) < probability:
        p = math.nextafter(p, math.inf)
    tail_count = (len(values) + 1) * Fraction(alpha) / (2 * family)
    if tail_count < 1:
        return _BootstrapResult(se, p, None, None, valid, _reason("bootstrap_tail_resolution"))
    ordered = np.sort(roots)
    tail = math.floor(tail_count) - 1
    lb = _bootstrap_bound(estimate, scale, float(ordered[-tail - 1]))
    ub = _bootstrap_bound(estimate, scale, float(ordered[tail]))
    if not math.isfinite(lb) or not math.isfinite(ub):
        return _BootstrapResult(None, None, None, None, valid, _reason("nonfinite_statistic"))
    return _BootstrapResult(se, p, lb, ub, valid, None)


class _ClusterTables(NamedTuple):
    groups: tuple[GroupEffect, ...]
    clan: tuple[ClanRow, ...]
    autoc: float
    qini: float
    rank_scales: tuple[float | None, float | None]


def _source_count(ids: np.ndarray) -> int:
    """Distinct sources, counted up to two: no support rule needs more."""
    if ids.dtype.kind not in "biuSUO":
        return min(int(np.unique(ids).size), 2)
    return 0 if not ids.size else 1 if bool(np.all(ids == ids[0])) else 2


def _support_failures(
    ids: np.ndarray,
    d: np.ndarray,
    arm_summary: Literal["welch", "score"],
    *,
    bootstrap: bool = False,
    fold: int | None = None,
    require_arms: bool = False,
) -> tuple[ClusterSupportFailure, ...]:
    """Original uncertainty needs two sources; resampled means need one."""
    prefix = "bootstrap_" if bootstrap else ""
    stage: Literal["original", "bootstrap"] = "bootstrap" if bootstrap else "original"
    failures = []
    minimum = 1 if bootstrap else 2
    if arm_summary == "welch" or require_arms:
        for arm in (0, 1):
            k = _source_count(ids[d == arm])
            name = "empty_arm" if k == 0 else "insufficient_arm_clusters"
            if k < (minimum if arm_summary == "welch" else 1):
                failures.append(
                    ClusterSupportFailure(
                        reason=_reason(prefix + name),
                        stage=stage,
                        fold=fold,
                        arm=arm,
                    )
                )
    if arm_summary == "score" and _source_count(ids) < minimum:
        failures.append(
            ClusterSupportFailure(
                reason=_reason(prefix + "insufficient_clusters"),
                stage=stage,
                fold=fold,
            )
        )
    return tuple(failures)


def _count_support_failures(
    failures: Sequence[ClusterSupportFailure],
) -> tuple[ClusterSupportFailure, ...]:
    counts: dict[tuple[str, str, int | None, int | None], ClusterSupportFailure] = {}
    for failure in failures:
        key = (failure.reason, failure.stage, failure.fold, failure.arm)
        previous = counts.get(key)
        counts[key] = (
            failure
            if previous is None
            else previous.model_copy(update={"count": previous.count + failure.count})
        )
    return tuple(counts.values())


def _replicate_reason(reason: str | None) -> str | None:
    # Point replicates remain defined without their own variance estimate.
    return (
        None
        if reason in {_reason("degenerate_cluster_variance"), _reason("insufficient_clusters")}
        else reason
    )


def _cluster_tables(
    holdout: _Holdout,
    psi: np.ndarray,
    n_groups: int,
    arm_summary: Literal["welch", "score"],
    clan_plan: tuple[tuple[str, np.ndarray], ...],
) -> _ClusterTables:
    assert holdout.cluster_ids is not None
    weights = _target_weights(holdout.cluster_ids, holdout.cluster_weight)
    groups = _sorted_groups(
        holdout.score,
        holdout.y,
        holdout.d,
        psi,
        n_groups,
        0.0,
        arm_summary=arm_summary,
        weights=weights,
        cluster_ids=holdout.cluster_ids,
    )
    bins = _weighted_bins(holdout.score, weights, n_groups)
    clan = tuple(
        _clan_row(
            name,
            x,
            bins == n_groups - 1,
            bins == 0,
            0.0,
            weights=weights,
            cluster_ids=holdout.cluster_ids,
        )
        for name, x in clan_plan
    )
    autoc, qini, references = _rank_components(holdout.score, psi, weights)
    scales = tuple(_cluster_mean(values, weights, holdout.cluster_ids).se for values in references)
    return _ClusterTables(groups, clan, autoc, qini, (scales[0], scales[1]))


class _ReplicateFrame(NamedTuple):
    """Validation rows in ascending score order, with dense source-cluster and tie codes."""

    score: np.ndarray
    y: np.ndarray
    d: np.ndarray
    psi: np.ndarray
    clan: tuple[np.ndarray, ...]
    codes: np.ndarray
    sizes: np.ndarray
    blocks: np.ndarray
    block_scores: np.ndarray
    weighting: Literal["member_count", "equal"]
    ipw: bool


def _replicate_frame(
    holdout: _Holdout,
    psi: np.ndarray,
    plan: tuple[tuple[str, np.ndarray], ...],
    population: _ClusterPopulation,
    arm_summary: Literal["welch", "score"],
) -> _ReplicateFrame:
    codes = np.zeros(holdout.score.size, dtype=np.intp)
    for g, rows in enumerate(population.members):
        codes[rows] = g
    order = np.argsort(holdout.score, kind="stable")
    block_scores, blocks = np.unique(holdout.score[order], return_inverse=True)
    return _ReplicateFrame(
        score=holdout.score[order],
        y=holdout.y[order],
        d=holdout.d[order],
        psi=psi[order],
        clan=tuple(values[order] for _, values in plan),
        codes=codes[order],
        sizes=np.array([rows.size for rows in population.members]),
        blocks=blocks,
        block_scores=block_scores,
        weighting=holdout.cluster_weight,
        ipw=arm_summary == "welch" and holdout.psi_fn is _ipw_psi_fn,
    )


class _Replicate(NamedTuple):
    """Resampled statistics in validation order, or the draw's support failures."""

    failures: tuple[ClusterSupportFailure, ...]
    values: tuple[float | None, ...] = ()
    scales: tuple[float | None, ...] = ()
    reasons: tuple[str | None, ...] = ()
    support: tuple[tuple[ClusterSupportFailure, ...], ...] = ()


def _replicate_statistics(
    frame: _ReplicateFrame,
    copies: np.ndarray,
    n_groups: int,
    arm_summary: Literal["welch", "score"],
    *,
    rank_scales: bool = True,
) -> _Replicate:
    """Recompute every validation statistic for a draw of copies[g] of source cluster g.

    Each retained row stands for copies[g] identical rows in distinct instances:
    its target mass is multiplied by copies[g], and instance variances divide each
    code's squared total by its copies. This equals materializing the draw.
    """
    keep = copies[frame.codes] > 0
    codes, d = frame.codes[keep], frame.d[keep]
    failures = _support_failures(codes, d, arm_summary, bootstrap=True)
    if failures:
        return _Replicate(failures)
    mass = copies / frame.sizes if frame.weighting == "equal" else copies.astype(float)
    weights, y, blocks = mass[codes], frame.y[keep], frame.blocks[keep]
    psi = _ipw_psi(y, d, weights=weights) if frame.ipw else frame.psi[keep]
    block_mass = np.bincount(blocks, weights=weights, minlength=frame.block_scores.size)
    present = block_mass > 0
    edges = [
        _block_cut(frame.block_scores[present], block_mass[present], g / n_groups, codes.size)
        for g in range(1, n_groups)
    ]
    # Rows ascend by score, so each group is a contiguous run ending at its cut.
    ends = np.searchsorted(frame.score[keep], edges, side="right").tolist()
    members = [
        slice(start, end) for start, end in zip([0, *ends], [*ends, codes.size], strict=True)
    ]
    values: list[float | None] = []
    scales: list[float | None] = []
    reasons: list[str | None] = [None, None]
    support: list[tuple[ClusterSupportFailure, ...]] = [(), ()]
    for inside in members:
        group_codes, group_d = codes[inside], d[inside]
        summary = _cluster_effect(
            y[inside],
            group_d,
            psi[inside],
            weights[inside],
            group_codes,
            arm_summary,
            copies=copies,
        )
        group_support = _support_failures(group_codes, group_d, arm_summary, bootstrap=True)
        values.append(summary.value)
        scales.append(summary.se)
        reasons.append(
            group_support[0].reason if group_support else _replicate_reason(summary.reason)
        )
        support.append(group_support)
    top, bottom = members[-1], members[0]
    # One replicate counts once per reason even if both CLAN sides fail.
    clan_support = tuple(
        dict.fromkeys(
            failure
            for side in (bottom, top)
            for failure in _support_failures(codes[side], d[side], "score", bootstrap=True)
        )
    )
    for column in frame.clan:
        x = column[keep]
        summary = _cluster_contrast(
            x[top],
            x[bottom],
            weights[top],
            weights[bottom],
            codes[top],
            codes[bottom],
            copies=copies,
        )
        values.append(summary.value)
        scales.append(summary.se)
        reasons.append(
            clan_support[0].reason if clan_support else _replicate_reason(summary.reason)
        )
        support.append(clan_support)
    # Number the retained tied scores densely from the top score down.
    ascending = np.cumsum(present)
    ranked = int(ascending[-1]) - ascending[blocks]
    curve = _block_rank_curve(ranked, psi, weights)
    # Deletion replicates consume rank values only; draws also studentize them.
    reference_scales = (
        tuple(
            _cluster_mean(r, weights, codes, copies=copies).se
            for r in _rank_references(curve, ranked)
        )
        if rank_scales
        else (None, None)
    )
    return _Replicate(
        (),
        (curve.autoc, curve.qini, *values),
        (*reference_scales, *scales),
        tuple(reasons),
        tuple(support),
    )


def _resampled_holdout(holdout: _Holdout, rows: np.ndarray, instances: np.ndarray) -> _Holdout:
    """Resample evaluation arrays; preserve source identity separately from occurrences."""
    assert holdout.cluster_ids is not None
    return holdout._replace(
        score=holdout.score[rows],
        y=holdout.y[rows],
        d=holdout.d[rows],
        cols={},
        unit_ids=holdout.unit_ids[rows],
        cluster_ids=instances,
        deployment_source_ids=(
            holdout.cluster_ids
            if holdout.deployment_source_ids is None
            else holdout.deployment_source_ids
        )[rows],
        bootstrap_population=None,
    )


def _resampled_psi(
    holdout: _Holdout, psi: np.ndarray, rows: np.ndarray, arm_summary: Literal["welch", "score"]
) -> np.ndarray:
    if arm_summary == "welch" and holdout.psi_fn is _ipw_psi_fn:
        assert holdout.cluster_ids is not None
        return _ipw_psi(
            holdout.y,
            holdout.d,
            weights=_target_weights(holdout.cluster_ids, holdout.cluster_weight),
        )
    # Frozen per-unit DR or custom scores: no learner is called in a replicate.
    return psi[rows]


def _cluster_validation_deletions(
    frame: _ReplicateFrame, n_groups: int, arm_summary: Literal["welch", "score"]
) -> list[list[float | None]]:
    """Complete statistics with each source cluster deleted once; None if unsupported."""
    sources = frame.sizes.size
    deleted: list[list[float | None]] = [[] for _ in range(2 + n_groups + len(frame.clan))]
    for omitted in range(sources if sources >= 2 else 0):
        copies = np.ones(sources, dtype=np.intp)
        copies[omitted] = 0
        replicate = _replicate_statistics(frame, copies, n_groups, arm_summary, rank_scales=False)
        values = replicate.values or (None,) * len(deleted)
        for target, value in zip(deleted, values, strict=True):
            target.append(value)
    return deleted


def _cluster_validation(
    holdout: _Holdout,
    psi: np.ndarray,
    *,
    n_groups: int,
    alpha: float,
    evaluation_population: CateEvaluationPopulation | None,
    arm_summary: Literal["welch", "score"],
    split_caveat: str,
    population: str | None,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> CateValidation:
    assert holdout.cluster_ids is not None
    ids = holdout.cluster_ids
    k = int(np.unique(ids).size)
    weights = _target_weights(ids, holdout.cluster_weight)
    ate = _cluster_effect(holdout.y, holdout.d, psi, weights, ids, arm_summary)
    plan = _clan_plan(holdout.cols, holdout.covariates)
    point = _cluster_tables(holdout, psi, n_groups, arm_summary, plan)
    # Keep categorical levels fixed, but recompute membership and means per draw.
    samples: list[list[float | None]] = [[] for _ in range(2 + n_groups + len(plan))]
    scales: list[list[float | None]] = [[] for _ in samples]
    failures: list[str | None] = [None for _ in samples]
    rng = np.random.default_rng(bootstrap_seed)
    support_events: list[list[ClusterSupportFailure]] = [[] for _ in samples]
    original_support = _support_failures(ids, holdout.d, arm_summary)
    population_draw = holdout.bootstrap_population or _cluster_population(
        ids,
        holdout.d,
        stratified=arm_summary == "welch",
    )
    frame = _replicate_frame(holdout, psi, plan, population_draw, arm_summary)
    deleted = _cluster_validation_deletions(frame, n_groups, arm_summary)
    for _ in range(bootstrap_repetitions if k >= 2 else 0):
        draws = _cluster_resample(population_draw, rng)
        replicate = _replicate_statistics(
            frame, np.bincount(draws, minlength=frame.sizes.size), n_groups, arm_summary
        )
        if replicate.failures:
            for j, values in enumerate(samples):
                values.append(None)
                scales[j].append(None)
                failures[j] = replicate.failures[0].reason
                support_events[j].extend(replicate.failures)
            continue
        for j, (value, scale, reason, support) in enumerate(
            zip(
                replicate.values,
                replicate.scales,
                replicate.reasons,
                replicate.support,
                strict=True,
            )
        ):
            samples[j].append(value if reason is None else None)
            scales[j].append(scale)
            support_events[j].extend(support)
            if reason is not None:
                failures[j] = reason

    rank_reason = (
        original_support[0].reason
        if original_support
        else _reason("degenerate_rank_distribution")
        if np.unique(holdout.score).size < 2
        else None
    )

    def rank(value: float, index: int) -> RankTest:
        jackknife = _delete_cluster_summary(value, deleted[index], population_draw)
        result = _centered_bootstrap(
            value,
            samples[index],
            alpha,
            scale=point.rank_scales[index],
            scales=scales[index],
            reason=rank_reason or failures[index],
        )
        result = _jackknife_hull(result, jackknife, alpha)
        return RankTest(
            estimate=value if math.isfinite(value) else None,
            se=result.se,
            p_value=result.p_value,
            lb=result.lb,
            ub=result.ub,
            n_clusters=k,
            cluster_weight=holdout.cluster_weight,
            uncertainty_method="bootstrap-t+cluster-jackknife-t",
            covariance_method="rank-response-reference",
            reference_df=jackknife.df,
            unavailable_reason=result.reason,
            bootstrap_seed=bootstrap_seed,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_valid_repetitions=result.valid,
            support_failures=_count_support_failures((*original_support, *support_events[index])),
        )

    def updated(row: GroupEffect | ClanRow, index: int, family: int) -> dict[str, object]:
        value = row.effect if isinstance(row, GroupEffect) else row.diff
        jackknife = _delete_cluster_summary(value, deleted[index], population_draw)
        result = _centered_bootstrap(
            value,
            samples[index],
            alpha,
            scale=row.se,
            scales=scales[index],
            family=family,
            reason=row.unavailable_reason or failures[index],
        )
        result = _jackknife_hull(result, jackknife, alpha, family=family)
        return {
            "se": result.se,
            "lb": result.lb,
            "ub": result.ub,
            "uncertainty_method": "bootstrap-t+cluster-jackknife-t",
            "covariance_method": row.uncertainty_method,
            "reference_df": jackknife.df,
            "cluster_weight": holdout.cluster_weight,
            "unavailable_reason": result.reason,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_valid_repetitions": result.valid,
            "support_failures": _count_support_failures(support_events[index]),
        }

    autoc, qini = rank(point.autoc, 0), rank(point.qini, 1)
    return CateValidation(
        n_train=holdout.n_train,
        n_holdout=holdout.y.size,
        holdout_ate=_summary_estimate(ate, alpha),
        holdout_ate_se=ate.se,
        groups=tuple(
            g.model_copy(update=updated(g, 2 + i, n_groups)) for i, g in enumerate(point.groups)
        ),
        clan=tuple(
            c.model_copy(update=updated(c, 2 + n_groups + i, len(plan)))
            for i, c in enumerate(point.clan)
        ),
        autoc=autoc,
        qini=qini,
        alpha=alpha,
        passed=autoc.p_value is not None and autoc.p_value < alpha,
        evaluation_population=evaluation_population,
        population=population,
        split_caveat=split_caveat,
        n_clusters=k,
        cluster_weight=holdout.cluster_weight,
        uncertainty_method=ate.method,
        reference_df=ate.df,
        unavailable_reason=ate.reason,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_valid_repetitions=autoc.bootstrap_valid_repetitions,
        support_failures=autoc.support_failures,
    )


def _cluster_policy(
    holdout: _Holdout,
    psi: np.ndarray,
    fraction: float,
    arm_summary: Literal["welch", "score"],
    deploy_grain: Literal["unit", "cluster"] = "unit",
) -> tuple[Deployment, _ClusterSummary, _ClusterSummary]:
    assert holdout.cluster_ids is not None
    weights = _target_weights(holdout.cluster_ids, holdout.cluster_weight)
    deployment = _deployment(
        holdout.score,
        fraction,
        holdout.cluster_ids,
        holdout.cluster_weight,
        deploy_grain,
        source_ids=holdout.deployment_source_ids,
    )
    targeted = deployment.targeted
    value = _cluster_effect(
        holdout.y[targeted],
        holdout.d[targeted],
        psi[targeted],
        weights[targeted],
        holdout.cluster_ids[targeted],
        arm_summary,
    )
    toc = _cluster_contrast(
        psi[targeted],
        psi,
        weights[targeted],
        weights,
        holdout.cluster_ids[targeted],
        holdout.cluster_ids,
    )
    return deployment, value, toc


def _cluster_rule(
    holdout: _Holdout,
    validation: CateValidation,
    psi: np.ndarray,
    *,
    fraction: float,
    alpha: float,
    arm_summary: Literal["welch", "score"],
    intervention_grain: Literal["unit", "cluster"],
    required_columns: tuple[str, ...],
    deploy_grain: Literal["unit", "cluster"] = "unit",
) -> TargetingRule:
    assert holdout.cluster_ids is not None
    seed, repetitions = validation.bootstrap_seed, validation.bootstrap_repetitions
    assert seed is not None and repetitions is not None
    deployment, value, toc = _cluster_policy(holdout, psi, fraction, arm_summary, deploy_grain)
    samples: list[float | None] = []
    scales: list[float | None] = []
    failure = None
    targeted = deployment.targeted
    empty_policy = not targeted.any()
    original_support = tuple(
        dict.fromkeys(
            (
                *_support_failures(holdout.cluster_ids, holdout.d, arm_summary),
                *_support_failures(holdout.cluster_ids[targeted], holdout.d[targeted], arm_summary),
            )
        )
    )
    support_events = list(original_support)
    rng = np.random.default_rng(seed)
    population_draw = holdout.bootstrap_population or _cluster_population(
        holdout.cluster_ids, holdout.d, stratified=arm_summary == "welch"
    )
    deleted: list[float | None] = []
    for rows in _cluster_deletions(population_draw) if not original_support else ():
        reduced = _resampled_holdout(holdout, rows, holdout.cluster_ids[rows])
        assert reduced.cluster_ids is not None
        if _support_failures(reduced.cluster_ids, reduced.d, arm_summary, bootstrap=True):
            deleted.append(None)
            continue
        _, deletion, _ = _cluster_policy(
            reduced,
            _resampled_psi(reduced, psi, rows, arm_summary),
            fraction,
            arm_summary,
            deploy_grain,
        )
        deleted.append(deletion.value)
    for _ in range(0 if original_support else repetitions):
        rows, instances = _cluster_draw(population_draw, rng)
        support = _support_failures(
            holdout.cluster_ids[rows],
            holdout.d[rows],
            arm_summary,
            bootstrap=True,
        )
        if support:
            samples.append(None)
            scales.append(None)
            failure = support[0].reason
            support_events.extend(support)
            continue
        resampled = _resampled_holdout(holdout, rows, instances)
        replicate_deployment, replicate, _ = _cluster_policy(
            resampled,
            _resampled_psi(resampled, psi, rows, arm_summary),
            fraction,
            arm_summary,
            deploy_grain,
        )
        targeted = replicate_deployment.targeted
        support = _support_failures(
            holdout.cluster_ids[rows][targeted],
            resampled.d[targeted],
            arm_summary,
            bootstrap=True,
        )
        support_events.extend(support)
        reason = support[0].reason if support else _replicate_reason(replicate.reason)
        samples.append(replicate.value if reason is None else None)
        scales.append(replicate.se)
        if reason is not None:
            failure = reason
    original_reason = (
        _reason("empty_group")
        if empty_policy
        else original_support[0].reason
        if original_support
        else value.reason
    )
    uncertainty = _centered_bootstrap(
        value.value,
        samples,
        alpha,
        scale=value.se,
        scales=scales,
        reason=original_reason or failure,
    )
    jackknife = _delete_cluster_summary(value.value, deleted, population_draw)
    uncertainty = _jackknife_hull(uncertainty, jackknife, alpha)
    passed = validation.passed and (
        empty_policy or (value.value is not None and toc.value is not None)
    )
    # The same holdout gates and reports this policy: retain point-only reporting.
    return TargetingRule(
        fraction=fraction,
        recommendation="target" if passed else "simple",
        validation=validation,
        threshold=deployment.threshold if passed else None,
        policy_value=Estimate(value=value.value) if passed and value.value is not None else None,
        uplift_vs_average=Estimate(value=toc.value) if passed and toc.value is not None else None,
        population=validation.population,
        n_clusters=validation.n_clusters,
        **_policy_metadata(holdout, deployment, intervention_grain, deploy_grain, required_columns),
        uncertainty_method="bootstrap-t+cluster-jackknife-t",
        covariance_method=value.method,
        reference_df=jackknife.df,
        unavailable_reason=uncertainty.reason,
        bootstrap_seed=seed,
        bootstrap_repetitions=repetitions,
        bootstrap_valid_repetitions=uncertainty.valid,
        support_failures=_count_support_failures(support_events),
    )


def _cluster_net_benefits(
    score: np.ndarray,
    psi: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    ids: np.ndarray,
    folds: np.ndarray,
    fractions: tuple[float, ...],
    cost: float,
    weighting: Literal["member_count", "equal"],
    *,
    randomized_score: bool,
    deploy_grain: Literal["unit", "cluster"] = "unit",
    source_ids: np.ndarray | None = None,
) -> tuple[_ClusterSummary, ...]:
    weights = _target_weights(ids, weighting)
    contributions = np.zeros((len(fractions), score.size))
    # Preserve failed resamples; only the original point's callers must refuse.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for label in np.unique(folds):
            val = folds == label
            local_psi = (
                _ipw_psi(y[val], d[val], weights=weights[val]) if randomized_score else psi[val]
            )
            for j, fraction in enumerate(fractions):
                targeted = _deployment(
                    score[val],
                    fraction,
                    ids[val],
                    weighting,
                    deploy_grain,
                    source_ids=None if source_ids is None else source_ids[val],
                ).targeted
                contributions[j, val] = np.where(targeted, local_psi - cost, 0.0)
        return tuple(_cluster_mean(values, weights, ids) for values in contributions)


def _inner_budget_share(
    score: np.ndarray,
    ids: np.ndarray,
    folds: np.ndarray,
    fraction: float,
    weighting: Literal["member_count", "equal"],
    grain: Literal["unit", "cluster"],
) -> float:
    weights = _target_weights(ids, weighting)
    targeted = np.zeros(score.size, dtype=bool)
    for label in np.unique(folds):
        val = folds == label
        targeted[val] = _deployment(score[val], fraction, ids[val], weighting, grain).targeted
    return float(weights[targeted].sum() / weights.sum())


def _cluster_inner_table(  # noqa: PLR0913
    score: np.ndarray,
    psi: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    ids: np.ndarray,
    folds: np.ndarray,
    fractions: tuple[float, ...],
    cost: float,
    weighting: Literal["member_count", "equal"],
    *,
    randomized_score: bool,
    deploy_grain: Literal["unit", "cluster"] = "unit",
    best_index: int,
    alpha: float,
    seed: int,
    repetitions: int,
    arm_summary: Literal["welch", "score"],
) -> tuple[FractionScore, ...]:
    """Conditional bootstrap of frozen inner predictions, preserving fold strata."""
    points = _cluster_net_benefits(
        score,
        psi,
        y,
        d,
        ids,
        folds,
        fractions,
        cost,
        weighting,
        randomized_score=randomized_score,
        deploy_grain=deploy_grain,
    )
    if any(point.value is None for point in points):
        _raise("estimation.targeting.nonfinite_statistic")
    samples: list[list[float | None]] = [[] for _ in fractions]
    scales: list[list[float | None]] = [[] for _ in fractions]
    fold_labels = np.unique(folds)
    original_support = tuple(
        failure
        for label in fold_labels
        for failure in _support_failures(
            ids[folds == label],
            d[folds == label],
            arm_summary,
            fold=int(label),
            require_arms=randomized_score,
        )
    )
    support_events = list(original_support)
    failure = None
    rng = np.random.default_rng(seed)
    population_draw = _cluster_population(ids, d, stratified=arm_summary == "welch", folds=folds)
    deleted: list[list[float | None]] = [[] for _ in fractions]
    for rows in _cluster_deletions(population_draw) if not original_support else ():
        support = tuple(
            failure
            for label in fold_labels
            for failure in _support_failures(
                ids[rows][folds[rows] == label],
                d[rows][folds[rows] == label],
                arm_summary,
                bootstrap=True,
                require_arms=randomized_score,
            )
        )
        if support:
            for target in deleted:
                target.append(None)
            continue
        deletion = _cluster_net_benefits(
            score[rows],
            psi[rows],
            y[rows],
            d[rows],
            ids[rows],
            folds[rows],
            fractions,
            cost,
            weighting,
            randomized_score=randomized_score,
            deploy_grain=deploy_grain,
            source_ids=ids[rows],
        )
        for target, value in zip(deleted, deletion, strict=True):
            target.append(value.value)
    for _ in range(0 if original_support else repetitions):
        rows, instances = _cluster_draw(population_draw, rng)
        support = tuple(
            failure
            for label in fold_labels
            for failure in _support_failures(
                ids[rows][folds[rows] == label],
                d[rows][folds[rows] == label],
                arm_summary,
                bootstrap=True,
                fold=int(label),
                require_arms=randomized_score,
            )
        )
        if support:
            support_events.extend(support)
            failure = support[0].reason
            for sample, scale in zip(samples, scales, strict=True):
                sample.append(None)
                scale.append(None)
            continue
        values = _cluster_net_benefits(
            score[rows],
            psi[rows],
            y[rows],
            d[rows],
            instances,
            folds[rows],
            fractions,
            cost,
            weighting,
            randomized_score=randomized_score,
            deploy_grain=deploy_grain,
            source_ids=ids[rows],
        )
        for sample, scale, value in zip(samples, scales, values, strict=True):
            sample.append(value.value)
            scale.append(value.se)
    table = []
    for index, (fraction, point, sample) in enumerate(zip(fractions, points, samples, strict=True)):
        assert point.value is not None
        reason = original_support[0].reason if original_support else failure
        result = _centered_bootstrap(
            point.value,
            sample,
            alpha,
            scale=point.se,
            scales=scales[index],
            reason=reason,
        )
        jackknife = _delete_cluster_summary(point.value, deleted[index], population_draw)
        result = _jackknife_hull(result, jackknife, alpha)
        selected = len(fractions) > 1 and index == best_index
        lb, ub = (None, None) if selected else (result.lb, result.ub)
        table.append(
            FractionScore(
                fraction=fraction,
                achieved_fraction=_inner_budget_share(
                    score, ids, folds, fraction, weighting, deploy_grain
                ),
                n=score.size,
                n_clusters=int(np.unique(ids).size),
                net_benefit=Estimate(
                    value=point.value,
                    lb=lb,
                    ub=ub,
                    level=1 - alpha if lb is not None else None,
                    alpha=alpha if lb is not None else None,
                ),
                cluster_weight=weighting,
                uncertainty_method="bootstrap-t+cluster-jackknife-t",
                covariance_method="policy-response-reference",
                reference_df=jackknife.df,
                unavailable_reason=result.reason,
                bootstrap_seed=seed,
                bootstrap_repetitions=repetitions,
                bootstrap_valid_repetitions=result.valid,
                support_failures=_count_support_failures(support_events),
            )
        )
    return tuple(table)
