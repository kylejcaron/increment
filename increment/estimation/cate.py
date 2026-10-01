"""Conditional average treatment effects (CATE) from an interacted regression.

Lin's interacted regression (Lin 2013, *Annals of Applied Statistics* 7(1),
295-318): regress the outcome on treatment, a covariate basis, and their
interactions. The treatment coefficient at the design centre is the ATE;
the interaction coefficients are the effect modifiers. This module holds
the whole estimator: covariate spec, basis transform, Gram-matrix rank
pruning, HC2 or cluster-score sandwich inference, the joint Wald heterogeneity test,
opt-in ARD shrinkage, and CATE scoring/contrasts.

Every covariate must be strictly pre-exposure (a post-treatment one
opens a collider path and biases every reported effect, including the
ATE) and results are on the absolute outcome scale (differences in the
outcome's own units, not ratios).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import chi2, f

from increment._identity import canonical_id_strings
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._deployment import (
    ClusterScore,
    deployment_ids,
    pool_cluster_scores,
    resolve_deploy_grain,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.results import Estimate

_IMPUTE_HINT = "nulls must be resolved before estimation -- impute them with increment.impute"
_INTERACT = "d:"
# A row fit exactly by the design has HC2 weight e_i^2 / (1 - h_i) = 0/0.
_MAX_LEVERAGE = 1.0 - 1e-8
# The adjustment basis may claim at most a tenth of the sample.
# Applied as `width * DENOM > n` so the bound is exact at the boundary.
_MAX_WIDTH_DENOM = 10
# Automatic relevance determination on the interaction block: prior
# precision cap, coefficient-vector convergence tolerance, sweep ceiling.
_ARD_MAX_ALPHA = 1e8
_ARD_TOL = 1e-8
_ARD_MAX_ITER = 200


class Covariate(CodedModel, BaseModel):
    """One pre-exposure column to model effect heterogeneity on.

    ``kind`` picks the basis: ``"continuous"`` standardizes to mean 0 /
    sd 1; ``"categorical"`` one-hot encodes against its modal level.

    ``knots`` opts a continuous covariate into a piecewise-linear
    (hinge) basis, free to bend at every knot but still linear in the
    coefficients. An ``int`` places that many knots at the interior
    quantiles of the fitting column; a tuple gives explicit positions.
    Continuous-only; opt in per covariate, since each knot spends an
    interaction column and a Wald degree of freedom.

    The basis is additive: knots on ``spend`` alongside a categorical
    ``country`` give one spend curve plus a per-country offset, not a
    differently shaped curve per country.
    """

    # No shared frozen base exists outside semantics, whose _Base is
    # extra="forbid" (not frozen) and must not be imported here.
    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["continuous", "categorical"] = "continuous"
    knots: int | tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_knots(self) -> Covariate:
        if self.knots is None:
            return self
        if self.kind != "continuous":
            _raise("estimation.cate.covariate.categorical_so_knots", self=self.name)
        if isinstance(self.knots, int):
            if self.knots < 1:
                _raise(
                    "estimation.cate.covariate.needs_least_one", name=self.name, knots=self.knots
                )
            return self
        if not self.knots:
            _raise("estimation.cate.covariate.was_empty_knot", self=self.name)
        if not all(math.isfinite(k) for k in self.knots):
            _raise(
                "estimation.cate.covariate.non_finite_knots",
                name=self.name,
                knots=tuple(self.knots),
            )
        if any(b <= a for a, b in zip(self.knots, self.knots[1:], strict=False)):
            _raise(
                "estimation.cate.covariate.needs_strictly_increasing",
                name=self.name,
                knots=self.knots,
            )
        return self


class _ColumnTransform(BaseModel):
    """Transform state learned from the fitting sample for one covariate."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["continuous", "categorical"]
    mean: float | None = None  # continuous
    scale: float | None = None  # continuous, ddof=1
    knots: tuple[float, ...] = ()  # continuous, in the covariate's own units
    levels: tuple[str, ...] = ()  # categorical, [0] is the reference (modal) level

    @property
    def column_names(self) -> tuple[str, ...]:
        if self.kind == "continuous":
            return (self.name, *(f"{self.name}>k{i}" for i in range(1, len(self.knots) + 1)))
        return tuple(f"{self.name}={level}" for level in self.levels[1:])


def _column(cols: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Fetch and null-check one column, naming the column in every error."""
    if name not in cols:
        _raise("estimation.cate.covariate_missing_from", name=name, columns=sorted(cols))
    values = np.asarray(cols[name])
    if values.ndim != 1:
        _raise("estimation.cate.covariate_shape", name=name, shape=values.shape)
    if values.dtype.kind in "fc":
        if not np.isfinite(values).all():
            _raise("estimation.cate.covariate_nulls_non", name=name)
    elif values.dtype.kind == "O":
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
            _raise("estimation.cate.covariate_nulls", name=name)
    return values


def _knot_positions(x: np.ndarray, knots: int | tuple[float, ...] | None) -> tuple[float, ...]:
    """Knot positions for one continuous column, in the column's own units.

    An int asks for that many interior quantiles ``(1..k)/(k+1)``; a
    tuple is taken as given. Learned once here and stored on the
    transform - re-deriving knots from a scoring frame would be a
    different basis, making the fitted coefficients meaningless.
    """
    if knots is None:
        return ()
    if isinstance(knots, int):
        return tuple(float(k) for k in np.quantile(x, np.arange(1, knots + 1) / (knots + 1)))
    return knots


class DesignSpec(BaseModel):
    """The fitted covariate basis: one transform per covariate, in order.

    ``fit`` learns the moments and levels; ``transform`` applies them.  A spec
    is reusable, so the same basis scores held-out units consistently.
    """

    model_config = ConfigDict(frozen=True)

    transforms: tuple[_ColumnTransform, ...]

    @classmethod
    def fit(cls, cols: Mapping[str, np.ndarray], covariates: Sequence[Covariate]) -> DesignSpec:
        """Learn the basis for *covariates* from the columns in *cols*."""
        transforms: list[_ColumnTransform] = []
        for cov in covariates:
            values = _column(cols, cov.name)
            if cov.kind == "continuous":
                x = values.astype(float, copy=False)
                scale = float(x.std(ddof=1)) if x.size > 1 else 0.0
                if not scale > 0.0:
                    _raise("estimation.cate.design.covariate_zero_variance", name=cov.name)
                transforms.append(
                    _ColumnTransform(
                        name=cov.name,
                        kind="continuous",
                        mean=float(x.mean()),
                        scale=scale,
                        knots=_knot_positions(x, cov.knots),
                    )
                )
                continue
            levels, counts = np.unique(values.astype(str), return_counts=True)
            # Descending frequency, ties broken lexically: the modal level
            # becomes the reference, and the ordering is reproducible.
            order = sorted(range(len(levels)), key=lambda i: (-counts[i], levels[i]))
            if len(levels) < 2:
                _raise(
                    "estimation.cate.design.covariate_single_level", name=cov.name, level=levels[0]
                )
            transforms.append(
                _ColumnTransform(
                    name=cov.name,
                    kind="categorical",
                    levels=tuple(str(levels[i]) for i in order),
                )
            )
        return cls(transforms=tuple(transforms))

    def column_names(self) -> tuple[str, ...]:
        """Basis column names, in the order ``transform`` emits them."""
        return tuple(name for t in self.transforms for name in t.column_names)

    def transform(self, cols: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
        """Apply the fitted basis to *cols*.

        Continuous columns standardize with the fitted mean/scale, never
        moments recomputed from *cols* - centering is the estimator's job.
        Categorical levels unseen at fit time encode as all zeros (the
        reference level).
        """
        blocks: list[np.ndarray] = []
        n: int | None = None
        for t in self.transforms:
            values = _column(cols, t.name)
            if n is None:
                n = values.size
            elif values.size != n:
                _raise(
                    "estimation.cate.design.covariate_rows_expected",
                    name=t.name,
                    size=values.size,
                    n=n,
                )
            if t.kind == "continuous":
                assert t.mean is not None and t.scale is not None  # set by fit
                z = (values.astype(float, copy=False) - t.mean) / t.scale
                if not t.knots:
                    blocks.append(z[:, None])
                    continue
                # Hinges live on the standardized scale too, so every column of
                # the block is in sd units and the coefficients are comparable.
                blocks.append(
                    np.column_stack(
                        [z, *(np.maximum(z - (k - t.mean) / t.scale, 0.0) for k in t.knots)]
                    )
                )
                continue
            as_str = values.astype(str)
            blocks.append(
                np.column_stack([(as_str == level).astype(float) for level in t.levels[1:]])
            )
        if not blocks:
            rows = next(iter(cols.values()), np.empty(0))
            return np.empty((np.asarray(rows).size, 0)), ()
        return np.hstack(blocks), self.column_names()


def prune_gram(
    zz: np.ndarray,
    names: Sequence[str],
    *,
    protect: int = 2,
    tol: float = 1e-8,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    """Drop rank-deficient columns from a Gram matrix ``Z'Z``.

    Columns are visited left to right and kept only if their variance
    conditional on the already-kept ones exceeds ``tol`` times their own
    diagonal (incremental Cholesky), so an aggregate column is dropped
    when it equals the sum of its parts. The first *protect* columns
    must survive; a degenerate one means the design is broken upstream.
    """
    zz = np.asarray(zz, dtype=float)
    if zz.ndim != 2 or zz.shape[0] != zz.shape[1]:
        _raise("estimation.cate.zz_square_gram", shape=zz.shape)
    p = zz.shape[0]
    if len(names) != p:
        _raise("estimation.cate.names_entries_gram", count=len(names), p=p)

    chol = np.zeros((p, p))
    kept: list[int] = []
    pruned: list[str] = []
    for j in range(p):
        diag = zz[j, j]
        k = len(kept)
        if k:
            w = solve_triangular(chol[:k, :k], zz[kept, j], lower=True)
            conditional = diag - float(w @ w)
        else:
            w = None
            conditional = diag
        if diag > 0.0 and conditional > tol * diag:
            if w is not None:
                chol[k, :k] = w
            chol[k, k] = math.sqrt(conditional)
            kept.append(j)
            continue
        if j < protect:
            _raise(
                "estimation.cate.protected_column_collinear",
                name=names[j],
                conditional=conditional,
                diag=diag,
            )
        pruned.append(names[j])
    index = np.asarray(kept, dtype=int)
    return zz[np.ix_(index, index)], tuple(kept), tuple(pruned)


class WaldTest(BaseModel):
    """Wald quadratic statistic and tested dimension.

    Clustered fits use F(statistic / df, df, fit.reference_df); unclustered
    fits use chi-square(df). Reference degrees are stored on the fit.
    """

    model_config = ConfigDict(frozen=True)

    statistic: float
    df: int
    p_value: float


class InteractionEffect(BaseModel):
    """One surviving interaction coefficient, on the absolute outcome scale.

    ``coef`` is the change in the treatment effect per unit of the basis
    column - one SD for a continuous covariate, or the step from the
    reference level for a one-hot column. Always the unpenalized
    least-squares coefficient; ``se``/``lb``/``ub`` use the fitted
    HC2 or cluster-score sandwich and its reference degrees of freedom.

    ``ard_coef`` is the same coefficient after ``fit_cate(ard=True)``'s
    evidence-maximizing shrinkage, ``None`` when not requested. It has
    no interval of its own - the unpenalized coefficient beside it
    carries the only interval on this row.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    coef: float
    se: float
    lb: float
    ub: float
    ard_coef: float | None = None


class CateScoreState(CodedModel, BaseModel):
    """Immutable portable scoring basis, fitted centering and effect coefficients."""

    model_config = ConfigDict(frozen=True)

    basis: DesignSpec
    means: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _aligned(self) -> CateScoreState:
        width = len(self.basis.column_names())
        if (
            len({t.name for t in self.basis.transforms}) != len(self.basis.transforms)
            or len(self.means) != width
            or len(self.coefficients) != width
            or not all(math.isfinite(v) for v in (*self.means, *self.coefficients))
        ):
            _raise("estimation.cate.score_state")
        for transform in self.basis.transforms:
            if transform.kind == "continuous" and (
                transform.mean is None
                or not math.isfinite(transform.mean)
                or transform.scale is None
                or not math.isfinite(transform.scale)
                or transform.scale <= 0
                or not all(math.isfinite(k) for k in transform.knots)
            ):
                _raise("estimation.cate.score_state")
            if transform.kind == "categorical" and (
                len(transform.levels) < 2 or len(set(transform.levels)) != len(transform.levels)
            ):
                _raise("estimation.cate.score_state")
        return self

    def _row_count(self, cols: Mapping[str, np.ndarray]) -> int:
        names = tuple(t.name for t in self.basis.transforms)
        if not names:
            names = tuple(cols)[:1]
        n = None
        for name in names:
            values = _column(cols, name)
            if n is not None and values.size != n:
                _raise(
                    "estimation.cate.design.covariate_rows_expected",
                    name=name,
                    size=values.size,
                    n=n,
                )
            n = values.size
        return 0 if n is None else n

    def _score_for_roster(
        self, cols: Mapping[str, np.ndarray], cluster_ids: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Use the roster to size constant predictions when no columns are needed."""
        roster_only = not self.basis.transforms and not cols and cluster_ids is not None
        n = np.asarray(cluster_ids).size if roster_only else self._row_count(cols)
        ids = deployment_ids(cluster_ids, n)
        score = np.full(n, self.intercept) if roster_only else self.score(cols)
        return score, ids

    def score(self, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """Apply the frozen basis; never relearn moments, knots or categories."""
        basis, _ = self.basis.transform(cols)
        out = np.full(basis.shape[0], self.intercept)
        for j, coef in enumerate(self.coefficients):
            if coef != 0:
                out += coef * (basis[:, j] - self.means[j])
        if not np.isfinite(out).all():
            _raise("estimation.cate.score_nonfinite")
        return out


CATE_OUTCOME_NULLS_NON = RefusalSpec(
    "estimation.cate.outcome_nulls_non",
    InvalidRequestError,
    lambda **_: f"outcome y has nulls or non-finite values; {_IMPUTE_HINT}",
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.cate.covariate.categorical_so_knots": "covariate {self!r} is categorical, so knots do not apply: a one-hot basis already fits each level freely. Drop knots or declare it continuous",
        "estimation.cate.outcome_nulls_non": CATE_OUTCOME_NULLS_NON,
        "estimation.cate.covariate.needs_least_one": "covariate {name!r} needs at least one knot, got {knots}",
        "estimation.cate.covariate.was_empty_knot": "covariate {self!r} was given an empty knot tuple; pass None for no knots",
        "estimation.cate.covariate.non_finite_knots": "covariate {name!r} has non-finite knots {knots}",
        "estimation.cate.covariate.needs_strictly_increasing": "covariate {name!r} needs strictly increasing knots, got {knots}; repeated or unordered positions give duplicate hinge columns",
        "estimation.cate.covariate_missing_from": "covariate {name!r} is missing from the unit frame; available columns: {columns}",
        "estimation.cate.covariate_shape": "covariate {name!r} must be 1-d, got shape {shape}",
        "estimation.cate.covariate_nulls_non": RefusalSpec(
            "estimation.cate.covariate_nulls_non",
            InvalidRequestError,
            lambda *, name: f"covariate {name!r} has nulls or non-finite values; {_IMPUTE_HINT}",
        ),
        "estimation.cate.covariate_nulls": RefusalSpec(
            "estimation.cate.covariate_nulls",
            InvalidRequestError,
            lambda *, name: f"covariate {name!r} has nulls; {_IMPUTE_HINT}",
        ),
        "estimation.cate.design.covariate_zero_variance": "covariate {name!r} has zero variance in the fitting sample, so it cannot explain heterogeneity; drop it",
        "estimation.cate.design.covariate_single_level": "covariate {name!r} has zero variance in the fitting sample (single level {level!r}); drop it",
        "estimation.cate.design.covariate_rows_expected": "covariate {name!r} has {size} rows, expected {n}",
        "estimation.cate.zz_square_gram": "zz must be a square Gram matrix, got shape {shape}",
        "estimation.cate.names_entries_gram": "names has {count} entries for a {p}x{p} Gram matrix",
        "estimation.cate.protected_column_collinear": "protected column {name!r} is collinear with the columns before it (conditional variance {conditional:.3g} of diagonal {diag:.3g}); the design is degenerate",
        "estimation.cate.cate.level_contradicts_alpha": "level={level!r} contradicts alpha={alpha!r}: the interval confidence implied by that alpha is {expected!r}",
        "estimation.cate.cate.covariate_missing_from": "covariate {transform!r} is missing from the requested point; every interacted covariate that survived pruning is required: {names}",
        "estimation.cate.covariate_declared_both": "covariate {name!r} is declared both {prior_kind!r} and {kind!r}; pick one",
        "estimation.cate.covariate_declared_knots": "covariate {name!r} is declared with knots {prior_knots!r} and {knots!r}; pick one -- the adjustment and interaction blocks are built from a single basis per covariate",
        "estimation.cate.covariate_columns_rows": "covariate columns have {rows} rows, expected {n} to match y",
        "estimation.cate.thin_cells.one_hot_level_singleton": "one-hot level {name!r} holds {total} unit(s): a level that identifies a single row is fit exactly and its HC2 weight is 0/0. Pool it into another level or drop the covariate",
        "estimation.cate.thin_cells.one_hot_level_interacted_min": "one-hot level {name!r} holds {n_t} treated and {n_c} control unit(s); an interacted level needs at least 2 per arm. Below that the cell is fit exactly, the HC2 weight is 0/0, and the reported standard error is silently far too small. Pool the level or drop the interaction",
        "estimation.cate.design_row_leverage": "design row {row} has leverage {leverage:.6g}: it is fit exactly, so its HC2 weight is 0/0 and the reported standard error would be silently far too small. The columns pinning it are {culprits}; pool or drop them",
        "estimation.cate.outcome_shape": "outcome y must be 1-d, got shape {shape}",
        "estimation.cate.treatment_shape_expected": "treatment d has shape {d_shape}, expected {y_shape} to match y",
        "estimation.cate.treatment_binary": "treatment d must be binary (0/1); got {values}",
        "estimation.cate.each_arm_needs": "each arm needs at least 2 units; got {n_treated} treated and {n_control} control",
        "estimation.cate.cluster_ids_shape": "cluster_ids shape {shape} must match the {n} outcome rows",
        "estimation.cate.intervention_grain": "intervention_grain must be 'unit' or 'cluster', got {grain!r}",
        "estimation.cate.intervention_grain_without_cluster": "cluster intervention requires cluster identity and a cluster count",
        "estimation.cate.cluster_count": "cluster count {n_clusters} exceeds the {n} fitted units",
        "estimation.cate.outcome_constant_within": "outcome y is constant within both arms, so there is no sampling variation to report a standard error for",
        "estimation.cate.design_too_wide": RefusalSpec(
            "estimation.cate.design_too_wide",
            InvalidRequestError,
            lambda *, width, n: (
                f"design is too wide: {width} covariate columns (one-hot levels and "
                f"interactions counted individually) against {n} units breaks the "
                f"p/n <= 1/{_MAX_WIDTH_DENOM} limit. Drop covariates or pool levels"
            ),
        ),
        "estimation.cate.cluster_weight": "cluster_weight must be 'member_count' or 'equal', got {weighting!r}",
        "estimation.cate.equal_weighting_without_cluster": "cluster_weight='equal' requires cluster IDs",
        "estimation.cate.insufficient_clusters": "cluster uncertainty requires at least two declared clusters, got {count}",
        "estimation.cate.rank_deficient": "clustered design is rank deficient at dimension {dimension}",
        "estimation.cate.invalid_covariance": "cluster covariance is unavailable: {reason}",
        "estimation.cate.single_cluster_direction": "{query} depends on a direction supported by only one cluster",
        "estimation.cate.singular_wald": "interaction Wald test of dimension {dimension} has singular covariance; its uncertainty is unavailable",
        "estimation.cate.uncertainty_unavailable": "{query} has no finite positive uncertainty under the cluster sandwich",
        "estimation.cate.inference_metadata": "fit dimension, covariance and reference degrees must match the fitted design",
        "estimation.cate.score_state": "Scoring state requires aligned finite coefficients and a valid fitted basis",
        "estimation.cate.score_nonfinite": "Fitted scoring produced non-finite predictions for these covariates",
    },
)

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
CATE_TREATMENT_BINARY = _REFUSALS["estimation.cate.treatment_binary"]
_raise = raiser(_REFUSALS)


class CateResult(CodedModel, BaseModel):
    """A fitted Lin (2013) interacted regression.

    ``ate`` is the treatment coefficient at the design centre (the
    weighted mean of the fitted treatment effects); ``se`` is
    its sandwich SE. Without clusters, ``se_unadjusted`` is Welch's SE;
    with clusters it uses the same weights and cluster sandwich fitted on
    ``[1, d]``. ``se_reduction`` reads the width the adjustment bought.

    ``dimension``, ``n_clusters`` (all observed IDs), immutable ``vcov`` and
    ``reference_df`` are the stored inference contract used by projections.
    Cluster intervals use t(K-1); the interaction Wald quadratic uses
    F(q, K-1) after division by q. The reference is cluster-asymptotic.
    ``unadjusted_vcov`` stores the separate two-column fit's covariance.
    Unavailable required uncertainty refuses rather than returning a partial fit.

    The state behind ``cate``/``contrast``/``score`` lives on excluded
    fields: a dumped result is report-only, so those need the live
    object. ``ate``/``se``/``lb``/``ub``/``heterogeneity`` are always
    the unpenalized fit's; ``fit_cate(ard=True)`` only moves the
    interaction estimates, in ``beta_ard`` (``None`` if ARD did not run).
    """

    model_config = ConfigDict(frozen=True)

    ate: float
    se: float
    lb: float
    ub: float
    #: The reported interval confidence. Constrained because a public result
    #: object and its serialized metadata must not be able to misstate it:
    #: `level` is `1 - alpha` and rounds to exactly 1.0 for a small but valid
    #: alpha, so both are carried and both are bounded.
    level: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    n: int
    n_treated: int
    n_control: int
    n_clusters: int | None = Field(default=None, ge=1, strict=True)
    dimension: int = Field(ge=2, strict=True)
    reference_df: int = Field(ge=1, strict=True)
    cluster_weight: Literal["member_count", "equal"] = "member_count"
    intervention_grain: Literal["unit", "cluster"] = "unit"
    welch_se_unadjusted: float | None = Field(default=None, exclude=True, repr=False)
    heterogeneity: WaldTest
    interactions: tuple[InteractionEffect, ...]
    pruned: tuple[str, ...]

    beta: tuple[float, ...] = Field(exclude=True, repr=False)
    vcov: tuple[tuple[float, ...], ...] = Field(exclude=True, repr=False)
    unadjusted_vcov: tuple[tuple[float, ...], ...] | None = Field(
        default=None, exclude=True, repr=False
    )
    unsupported_directions: tuple[tuple[float, ...], ...] = Field(
        default=(), exclude=True, repr=False
    )
    columns: tuple[str, ...] = Field(exclude=True, repr=False)
    main_spec: DesignSpec = Field(exclude=True, repr=False)
    interaction_spec: DesignSpec = Field(exclude=True, repr=False)
    interaction_means: tuple[float, ...] = Field(exclude=True, repr=False)
    interaction_positions: tuple[tuple[int, int], ...] = Field(exclude=True, repr=False)
    beta_ard: tuple[float, ...] | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _level_matches_alpha(self) -> CateResult:
        """``level`` must be exactly the float64 value of ``1 - alpha``.

        Bounding the two fields independently still allowed a result to
        serialize contradictory confidence metadata (level=0.5 with alpha=0.05).
        The comparison is against ``fsum((1.0, -alpha))`` rather than ``1 -
        alpha`` literally, because that is the value the interval was built
        with -- and for a tiny alpha it rounds to exactly 1.0, which is why both
        fields are carried in the first place.
        """
        expected = math.fsum((1.0, -self.alpha))
        if self.level != expected:
            _raise(
                "estimation.cate.cate.level_contradicts_alpha",
                level=self.level,
                alpha=self.alpha,
                expected=expected,
            )
        return self

    @model_validator(mode="after")
    def _cluster_metadata(self) -> CateResult:
        if self.intervention_grain == "cluster" and self.n_clusters is None:
            _raise("estimation.cate.intervention_grain_without_cluster")
        if self.n_clusters is not None and self.n_clusters > self.n:
            _raise("estimation.cate.cluster_count", n_clusters=self.n_clusters, n=self.n)
        return self

    @model_validator(mode="after")
    def _inference_metadata(self) -> CateResult:
        if self.n_clusters is not None and self.n_clusters < 2:
            _raise("estimation.cate.insufficient_clusters", count=self.n_clusters)
        expected = self.n - self.dimension if self.n_clusters is None else self.n_clusters - 1
        if (
            self.dimension != len(self.columns)
            or self.dimension != len(self.beta)
            or self.reference_df != expected
            or len(self.vcov) != self.dimension
            or any(len(row) != self.dimension for row in self.vcov)
        ):
            _raise("estimation.cate.inference_metadata")
        if self.unsupported_directions and (
            any(len(direction) != self.dimension for direction in self.unsupported_directions)
            or not np.isfinite(self.unsupported_directions).all()
        ):
            _raise("estimation.cate.inference_metadata")
        if self.cluster_weight == "equal" and self.n_clusters is None:
            _raise("estimation.cate.equal_weighting_without_cluster")
        if self.n_clusters is not None:
            _check_covariance(np.asarray(self.vcov))
            if (
                self.unadjusted_vcov is None
                or len(self.unadjusted_vcov) != 2
                or any(len(row) != 2 for row in self.unadjusted_vcov)
            ):
                _raise("estimation.cate.inference_metadata")
            _check_covariance(np.asarray(self.unadjusted_vcov))
            _cluster_se(
                np.array([0.0, 1.0]), np.asarray(self.unadjusted_vcov), query="unadjusted ATE"
            )
        elif self.welch_se_unadjusted is None:
            _raise("estimation.cate.inference_metadata")
        return self

    @computed_field
    @property
    def se_unadjusted(self) -> float:
        """Read the cluster two-column covariance, or the unclustered Welch SE."""
        if self.n_clusters is not None:
            assert self.unadjusted_vcov is not None
            return _cluster_se(
                np.array([0.0, 1.0]), np.asarray(self.unadjusted_vcov), query="unadjusted ATE"
            )
        assert self.welch_se_unadjusted is not None
        return self.welch_se_unadjusted

    @property
    def se_reduction(self) -> float:
        """Fraction of the unadjusted standard error the adjustment removed."""
        reduction = 1.0 - self.se / self.se_unadjusted
        if self.n_clusters is not None and not math.isfinite(reduction):
            _raise("estimation.cate.uncertainty_unavailable", query="standard error reduction")
        return reduction

    def cate(self, values: Mapping[str, float | str]) -> Estimate:
        """The treatment effect at one point in covariate space.

        *values* must name every interacted covariate that kept a basis
        column (pruned columns drop out; extra entries are ignored).
        Under ``fit_cate(ard=True)`` the point is shrunken and ships
        without an interval - the unpenalized sandwich's half-width
        would assert a nominal coverage the shrunken point does not
        have, the same reason ``InteractionEffect.ard_coef`` carries no
        interval of its own.
        """
        return self._estimate(self._contrast_vector(values, self._kept_spec()))

    def contrast(self, a: Mapping[str, float | str], b: Mapping[str, float | str]) -> Estimate:
        """The difference in treatment effect between two covariate points.

        The treatment entries cancel, leaving a pure heterogeneity
        contrast that may cross covariates (e.g. ``ios`` at high spend
        versus ``android`` at low spend). The ARD caveat on ``cate``
        applies identically.
        """
        spec = self._kept_spec()
        return self._estimate(self._contrast_vector(a, spec) - self._contrast_vector(b, spec))

    @property
    def score_state(self) -> CateScoreState:
        """Portable prediction state, also used by saved targeting policies."""
        spec = self._kept_spec()
        original_positions = self._spec_original_positions(spec)
        coefficients = [0.0] * len(original_positions)
        means = [self.interaction_means[j] for j in original_positions]
        by_original = {original: retained for retained, original in self.interaction_positions}
        values = {
            retained: (self.beta[retained] if self.beta_ard is None else self.beta_ard[retained])
            for retained, _ in self.interaction_positions
        }
        for local, original in enumerate(original_positions):
            retained = by_original.get(original)
            if retained is not None:
                coefficients[local] = values[retained]
        return CateScoreState(
            basis=spec,
            means=tuple(means),
            coefficients=tuple(coefficients),
            intercept=self.ate,
        )

    @overload
    def score(
        self,
        cols: Mapping[str, np.ndarray],
        *,
        cluster_ids: np.ndarray | None = None,
        deploy_grain: Literal["unit"],
    ) -> np.ndarray: ...

    @overload
    def score(
        self,
        cols: Mapping[str, np.ndarray],
        *,
        cluster_ids: np.ndarray | None = None,
        deploy_grain: Literal["cluster"],
    ) -> tuple[ClusterScore, ...]: ...

    @overload
    def score(
        self,
        cols: Mapping[str, np.ndarray],
        *,
        cluster_ids: np.ndarray | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
    ) -> np.ndarray | tuple[ClusterScore, ...]: ...

    def score(
        self,
        cols: Mapping[str, np.ndarray],
        *,
        cluster_ids: np.ndarray | None = None,
        deploy_grain: Literal["unit", "cluster"] | None = None,
    ) -> np.ndarray | tuple[ClusterScore, ...]:
        """Score units or return immutable keyed cluster means in canonical ID order.

        The default follows the declared intervention, never dependence IDs.
        Cluster deployment requires aligned IDs; a cluster intervention cannot
        request unit deployment. Internal evaluation uses ``score_state.score``
        to retain aligned unit predictions before applying a deployment budget.
        Scores use the fitted (possibly ARD-shrunken) model, not new-data moments.
        """
        grain = resolve_deploy_grain(
            self.intervention_grain, deploy_grain, clustered=cluster_ids is not None
        )
        state = self.score_state
        score, ids = state._score_for_roster(cols, cluster_ids)
        if grain == "unit":
            return score
        assert ids is not None
        return pool_cluster_scores(score, ids)

    def _kept_spec(self) -> DesignSpec:
        """Interaction transforms that kept at least one basis column."""
        kept_original = {original for _, original in self.interaction_positions}
        transforms: list[_ColumnTransform] = []
        offset = 0
        for transform in self.interaction_spec.transforms:
            width = len(transform.column_names)
            if kept_original.intersection(range(offset, offset + width)):
                transforms.append(transform)
            offset += width
        return DesignSpec(transforms=tuple(transforms))

    def _spec_original_positions(self, spec: DesignSpec) -> tuple[int, ...]:
        """Original interaction-basis positions emitted by a retained spec."""
        selected = {transform.name for transform in spec.transforms}
        positions: list[int] = []
        offset = 0
        for transform in self.interaction_spec.transforms:
            width = len(transform.column_names)
            if transform.name in selected:
                positions.extend(range(offset, offset + width))
            offset += width
        return tuple(positions)

    def _centered_basis(self, spec: DesignSpec, cols: Mapping[str, np.ndarray]) -> np.ndarray:
        """Apply *spec*, centered at fitted means by structural position."""
        basis, _ = spec.transform(cols)
        positions = self._spec_original_positions(spec)
        means = np.array([self.interaction_means[j] for j in positions])
        return basis - means

    def _contrast_vector(self, values: Mapping[str, float | str], spec: DesignSpec) -> np.ndarray:
        """Loading vector for the treatment effect at one point of *spec*."""
        a = np.zeros(self.dimension)
        a[1] = 1.0  # the treatment column; fit_cate protects position 1
        if not self.interactions:
            return a
        point: dict[str, np.ndarray] = {}
        for transform in spec.transforms:
            if transform.name not in values:
                _raise(
                    "estimation.cate.cate.covariate_missing_from",
                    transform=transform.name,
                    names=[t.name for t in spec.transforms],
                )
            point[transform.name] = np.asarray([values[transform.name]])
        row = self._centered_basis(spec, point)[0]
        local = dict(zip(self._spec_original_positions(spec), range(row.size), strict=True))
        for retained, original in self.interaction_positions:
            a[retained] = row[local[original]]
        return a

    def _estimate(self, a: np.ndarray) -> Estimate:
        if self.n_clusters is not None:
            _refuse_cluster_direction(a, self.unsupported_directions, query="CATE contrast")
        if self.beta_ard is not None:
            # Shrunken point, no interval: the unpenalized sandwich's
            # half-width would assert a coverage the shrunken point does
            # not have (see `cate`'s docstring; mirrors `ard_coef`).
            return Estimate(value=float(a @ np.asarray(self.beta_ard)))
        value = float(a @ np.asarray(self.beta))
        if self.n_clusters is None:
            se = math.sqrt(max(float(a @ np.asarray(self.vcov) @ a), 0.0))
        else:
            se = _cluster_se(a, np.asarray(self.vcov), query="CATE contrast")
        dof = self.reference_df
        t_crit = two_sided_critical_value(
            student_t_isf, self.alpha, dof, what="CATE point estimate"
        )
        half = t_crit * se
        if self.n_clusters is not None:
            _check_cluster_interval(value, half, query="CATE contrast")
        return Estimate(
            value=value, lb=value - half, ub=value + half, level=self.level, alpha=self.alpha
        )


def _dedup(covariates: Sequence[Covariate]) -> tuple[Covariate, ...]:
    """First occurrence of each covariate name, order preserved."""
    seen: dict[str, Covariate] = {}
    for cov in covariates:
        prior = seen.get(cov.name)
        if prior is None:
            seen[cov.name] = cov
        elif prior.kind != cov.kind:
            _raise(
                "estimation.cate.covariate_declared_both",
                name=cov.name,
                kind=cov.kind,
                prior_kind=prior.kind,
            )
        elif prior.knots != cov.knots:
            _raise(
                "estimation.cate.covariate_declared_knots",
                name=cov.name,
                knots=cov.knots,
                prior_knots=prior.knots,
            )
    return tuple(seen.values())


def _basis(spec: DesignSpec, cols: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    """Untransformed-width-safe basis: an empty spec still yields *n* rows."""
    if not spec.transforms:
        return np.empty((n, 0))
    block, _ = spec.transform(cols)
    if block.shape[0] != n:
        _raise("estimation.cate.covariate_columns_rows", rows=block.shape[0], n=n)
    return block


def _one_hot_flags(spec: DesignSpec) -> tuple[bool, ...]:
    """Per basis column: whether it is a one-hot level indicator."""
    return tuple(t.kind == "categorical" for t in spec.transforms for _ in t.column_names)


def _refuse_thin_cells(
    kept: Sequence[int],
    names: Sequence[str],
    main_raw: np.ndarray,
    main_flags: Sequence[bool],
    interact_raw: np.ndarray,
    interact_flags: Sequence[bool],
    d: np.ndarray,
) -> None:
    """Refuse a retained one-hot level too sparse for a usable sandwich.

    An interacted level with fewer than 2 units per arm (or an
    uninteracted level with fewer than 2 units total) is fit exactly:
    leverage goes to 1, the HC2 weight to ``0/0``, and the reported
    standard error lands far below the true sampling error with no
    warning. Rank pruning and the p/n guard do not catch this. Only
    *kept* columns are checked.
    """
    treated = d == 1.0
    offset = 2 + main_raw.shape[1]
    for position in kept:
        if position < 2:
            continue
        interacted = position >= offset
        j = position - offset if interacted else position - 2
        if not (interact_flags[j] if interacted else main_flags[j]):
            continue
        present = (interact_raw[:, j] if interacted else main_raw[:, j]) != 0.0
        if not interacted:
            total = int(present.sum())
            if total < 2:
                _raise(
                    "estimation.cate.thin_cells.one_hot_level_singleton",
                    name=names[position],
                    total=total,
                )
            continue
        n_t = int((present & treated).sum())
        n_c = int(present.sum()) - n_t
        if n_t < 2 or n_c < 2:
            _raise(
                "estimation.cate.thin_cells.one_hot_level_interacted_min",
                name=names[position],
                n_t=n_t,
                n_c=n_c,
            )


def _refuse_exact_fits(
    leverage: np.ndarray, z: np.ndarray, xtx_inv: np.ndarray, names: Sequence[str]
) -> None:
    """Backstop on the hat diagonal: no row may be fit exactly.

    ``_refuse_thin_cells`` catches the one-hot case by name before the fit;
    this catches every other way a design can pin a single row, including
    continuous columns that happen to isolate one unit.
    """
    row = int(np.argmax(leverage))
    if leverage[row] < _MAX_LEVERAGE:
        return
    # Per-column shares of h_i, which sum to h_i exactly.
    share = z[row] * (xtx_inv @ z[row])
    culprits = [names[j] for j in np.argsort(share)[::-1][:3] if share[j] > 0.0]
    _raise(
        "estimation.cate.design_row_leverage",
        row=row,
        leverage=float(leverage[row]),
        culprits=culprits,
    )


def _ard_interactions(
    other: np.ndarray, interactions: np.ndarray, y: np.ndarray, sigma_sq: float
) -> np.ndarray:
    """Evidence-maximized interaction coefficients, the other blocks partialled out.

    Automatic relevance determination in MacKay's form (MacKay 1992,
    *Neural Computation* 4(3), 415-447): one Gaussian prior precision
    ``alpha_j`` per interaction column, chosen to maximize the marginal
    likelihood, so a wide interaction block tunes its own effective
    width instead of spending a degree of freedom on every column.

    The interaction block is the only penalized block; *interactions*
    and *y* are residualized on *other* first (Frisch-Waugh-Lovell), so
    at ``alpha = 0`` the sweep reproduces the OLS coefficients exactly.
    Noise is held fixed at the supplied variance *sigma_sq*, so
    nothing here moves the ATE, the sandwich, or the joint Wald test.

    Each sweep re-solves the posterior before updating ``alpha``::

        Sigma    = sigma^2 (Z'Z + sigma^2 diag(alpha))^-1
        mu       = Sigma Z'r / sigma^2
        alpha_j <- (1 - alpha_j Sigma_jj) / mu_j^2

    Re-solving first keeps the fixed point stable; updating ``alpha``
    against a fixed least-squares coefficient instead diverges for a
    null column. Convergence has a slow tail near the relevance
    boundary, so a minority of fits reach ``_ARD_MAX_ITER`` without
    meeting ``_ARD_TOL`` - not a failure: a hundredfold ceiling moves
    the largest coefficient by 1e-4 on the calibration suite's design.
    """
    q, _ = np.linalg.qr(other)
    z = interactions - q @ (q.T @ interactions)
    r = y - q @ (q.T @ y)
    ztz = z.T @ z
    ztr = z.T @ r
    if not sigma_sq > 0.0:
        # A design that fits y exactly leaves the evidence nothing to trade
        # relevance against; the sigma^2 -> 0 limit of the sweep is plain OLS.
        return np.linalg.solve(ztz, ztr)

    width = z.shape[1]
    eye = np.eye(width)
    alpha = np.ones(width)
    mu = np.zeros(width)
    for _ in range(_ARD_MAX_ITER):
        posterior = sigma_sq * cho_solve(
            cho_factor(ztz + sigma_sq * np.diag(alpha), lower=True), eye
        )
        previous, mu = mu, posterior @ ztr / sigma_sq
        if float(np.max(np.abs(mu - previous))) < _ARD_TOL:
            break
        # Effective degrees of freedom column j claims, in (0, 1]: Z'Z >= 0
        # bounds Sigma_jj by 1 / alpha_j (equality only where Z is blind).
        claimed = 1.0 - alpha * np.diag(posterior)
        # Flooring mu_j^2 at claimed_j / _ARD_MAX_ALPHA is the alpha cap,
        # keeping a numerically dead column out of a 0 / 0.
        alpha = claimed / np.maximum(mu**2, claimed / _ARD_MAX_ALPHA)
    return mu


def _cluster_index(
    cluster_ids: np.ndarray | None, n: int, intervention_grain: Literal["unit", "cluster"]
) -> tuple[np.ndarray, np.ndarray] | None:
    if intervention_grain not in ("unit", "cluster"):
        _raise("estimation.cate.intervention_grain", grain=intervention_grain)
    if cluster_ids is None:
        if intervention_grain == "cluster":
            _raise("estimation.cate.intervention_grain_without_cluster")
        return None
    ids = np.asarray(cluster_ids, dtype=object)
    if ids.shape != (n,):
        _raise("estimation.cate.cluster_ids_shape", shape=ids.shape, n=n)
    _, groups, sizes = np.unique(
        canonical_id_strings(ids, what="cluster_ids"), return_inverse=True, return_counts=True
    )
    return groups, sizes


def _validate_cluster_weight(weighting: str, *, clustered: bool) -> None:
    if weighting not in ("member_count", "equal"):
        _raise("estimation.cate.cluster_weight", weighting=weighting)
    if weighting == "equal" and not clustered:
        _raise("estimation.cate.equal_weighting_without_cluster")


def _check_covariance(covariance: np.ndarray) -> None:
    """Accept PSD (including singular) covariance, within roundoff only."""
    if (
        covariance.ndim != 2
        or covariance.shape[0] == 0
        or covariance.shape[0] != covariance.shape[1]
    ):
        _raise("estimation.cate.invalid_covariance", reason="expected a nonempty square matrix")
    if not np.isfinite(covariance).all():
        _raise("estimation.cate.invalid_covariance", reason="nonfinite entries")
    scale = float(np.max(np.abs(covariance)))
    tolerance = 64 * np.finfo(float).eps * covariance.shape[0] * scale
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        _raise("estimation.cate.invalid_covariance", reason="asymmetric matrix")
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        _raise("estimation.cate.invalid_covariance", reason="eigendecomposition failed")
    if eigenvalues[0] < -tolerance:
        _raise("estimation.cate.invalid_covariance", reason="matrix is not positive semidefinite")


def _refuse_cluster_direction(
    loading: np.ndarray, directions: Sequence[Sequence[float]], *, query: str
) -> None:
    """A contrast must annihilate every coefficient direction pinned by one cluster."""
    if directions and np.any(
        np.abs(np.asarray(directions) @ loading) > 1e-9 * np.linalg.norm(loading)
    ):
        _raise("estimation.cate.single_cluster_direction", query=query)


def _cluster_se(loading: np.ndarray, covariance: np.ndarray, *, query: str) -> float:
    if not np.any(loading):
        return 0.0  # The identically zero contrast is known without uncertainty.
    variance = float(loading @ covariance @ loading)
    if not math.isfinite(variance) or variance <= 0.0:
        _raise("estimation.cate.uncertainty_unavailable", query=query)
    return math.sqrt(variance)


def _check_cluster_interval(value: float, half: float, *, query: str) -> None:
    if not all(math.isfinite(v) for v in (value, half, value - half, value + half)):
        _raise("estimation.cate.uncertainty_unavailable", query=query)


def _refuse_cluster_roundoff_residuals(
    z: np.ndarray,
    centered_y: np.ndarray,
    weights: np.ndarray,
    beta: np.ndarray,
    inverse: np.ndarray,
    residual: np.ndarray,
) -> None:
    """Refuse residuals explained entirely by normal-equation roundoff.

    Absolute products bound errors in Z'Wy and Z'WZ beta; |(Z'WZ)^-1|
    propagates them to coefficients, then |Z| to fitted outcomes. Include
    subtraction and prediction error too. Centered outcomes keep the bound
    independent of the intercept offset, with no absolute outcome-scale floor.
    """
    absolute_z = np.abs(z)
    outcome_scale = np.abs(centered_y) + absolute_z @ np.abs(beta)
    normal_scale = absolute_z.T @ (weights * outcome_scale)
    propagated_scale = absolute_z @ (np.abs(inverse) @ normal_scale)
    # Cover accumulation, Gram factorization, solves, and prediction together.
    rounding = 8 * max(z.shape) * np.finfo(float).eps
    bound = (rounding / (1 - rounding)) * (outcome_scale + propagated_scale)
    if np.all(np.abs(residual) <= bound):
        _raise("estimation.cate.uncertainty_unavailable", query="cluster residuals")


def _cluster_sandwich(
    z: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...]]:
    """Weighted CR1/HC0 fit; every declared cluster enters the score and count."""
    weighted = np.sqrt(weights)[:, None] * z
    dimension = z.shape[1]
    if not np.isfinite(weighted).all():
        _raise("estimation.cate.rank_deficient", dimension=dimension)
    try:
        if np.linalg.matrix_rank(weighted) != dimension:
            _raise("estimation.cate.rank_deficient", dimension=dimension)
        gram = weighted.T @ weighted
        if not np.isfinite(gram).all():
            _raise("estimation.cate.invalid_covariance", reason="nonfinite design Gram matrix")
        lower = np.linalg.cholesky(gram)
        inverse = cho_solve((lower, True), np.eye(dimension))
    except np.linalg.LinAlgError:
        _raise("estimation.cate.rank_deficient", dimension=dimension)
    # Fit differences so the intercept's offset cannot erase treatment effects.
    centered_y = y - y[0]
    beta = inverse @ (z.T @ (weights * centered_y))
    residual = centered_y - z @ beta
    _refuse_cluster_roundoff_residuals(z, centered_y, weights, beta, inverse, residual)
    beta[0] = math.fsum((float(beta[0]), float(y[0])))
    scores = np.zeros((count, dimension))
    np.add.at(scores, groups, (weights * residual)[:, None] * z)
    influence = scores @ inverse
    covariance = (count / (count - 1)) * (influence.T @ influence)
    _check_covariance(covariance)

    # Eigenvalue one in L^-1 H_g L^-T identifies a direction unique to g.
    # This diagnoses identification only; it never rescales cluster scores.
    normalized = solve_triangular(lower, weighted.T, lower=True).T
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    directions: list[tuple[float, ...]] = []
    tolerance = 128 * np.finfo(float).eps * max(z.shape)
    for rows in np.split(order, boundaries):
        block = normalized[rows]
        try:
            eigenvalues, vectors = np.linalg.eigh(block.T @ block)
        except np.linalg.LinAlgError:
            _raise(
                "estimation.cate.invalid_covariance", reason="cluster identification check failed"
            )
        for vector in vectors[:, eigenvalues >= 1.0 - tolerance].T:
            direction = solve_triangular(lower.T, vector, lower=False)
            direction /= np.linalg.norm(direction)
            directions.append(tuple(float(v) for v in direction))
    return beta, covariance, tuple(directions)


def _cluster_ard_variance(
    other: np.ndarray, interactions: np.ndarray, covariance: np.ndarray
) -> float:
    q, _ = np.linalg.qr(other)
    residualized = interactions - q @ (q.T @ interactions)
    gram = residualized.T @ residualized
    # Directional-average precision approximation, not correlated-likelihood ARD.
    sigma_sq = float(np.trace(gram @ covariance)) / interactions.shape[1]
    if not math.isfinite(sigma_sq) or sigma_sq <= 0.0:
        _raise("estimation.cate.uncertainty_unavailable", query="cluster ARD precision")
    return sigma_sq


def _cluster_wald(beta: np.ndarray, covariance: np.ndarray, reference_df: int) -> WaldTest:
    dimension = beta.size
    _check_covariance(covariance)
    try:
        if np.linalg.matrix_rank(covariance) != dimension:
            _raise("estimation.cate.singular_wald", dimension=dimension)
        factor = cho_factor(covariance, lower=True)
        statistic = float(beta @ cho_solve(factor, beta))
    except np.linalg.LinAlgError:
        _raise("estimation.cate.singular_wald", dimension=dimension)
    if not math.isfinite(statistic) or statistic < 0.0:
        _raise("estimation.cate.uncertainty_unavailable", query="interaction Wald test")
    return WaldTest(
        statistic=statistic,
        df=dimension,
        p_value=float(f.sf(statistic / dimension, dimension, reference_df)),
    )


def _fit_hc2(
    z: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    names: tuple[str, ...],
    main_raw: np.ndarray,
    main_spec: DesignSpec,
    interact_raw: np.ndarray,
    interact_spec: DesignSpec,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    gram, kept, pruned = prune_gram(z.T @ z, names, protect=2)
    _refuse_thin_cells(
        kept,
        names,
        main_raw,
        _one_hot_flags(main_spec),
        interact_raw,
        _one_hot_flags(interact_spec),
        d,
    )
    zk = z[:, list(kept)]
    kept_names = tuple(names[i] for i in kept)
    xtx_inv = cho_solve(cho_factor(gram, lower=True), np.eye(gram.shape[0]))
    beta = xtx_inv @ (zk.T @ y)
    resid = y - zk @ beta
    # Row-wise contractions avoid forming an n x n hat matrix.
    leverage = np.einsum("ij,jk,ik->i", zk, xtx_inv, zk)
    _refuse_exact_fits(leverage, zk, xtx_inv, kept_names)
    meat = np.einsum("ij,ik,i->jk", zk, zk, resid**2 / (1.0 - leverage))
    return beta, xtx_inv @ meat @ xtx_inv, resid, zk, kept, pruned, kept_names


def _shrink_interaction_block(
    beta: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    positions: list[int],
    covariance: np.ndarray,
    weights: np.ndarray,
    *,
    residual_variance: float | None,
) -> tuple[float, ...]:
    interacted = set(positions)
    other_positions = [j for j in range(z.shape[1]) if j not in interacted]
    shrunk = beta.copy()
    if residual_variance is not None:
        shrunk[positions] = _ard_interactions(
            z[:, other_positions], z[:, positions], y, residual_variance
        )
    else:
        root_weights = np.sqrt(weights)
        other = root_weights[:, None] * z[:, other_positions]
        interaction_block = root_weights[:, None] * z[:, positions]
        sigma_sq = _cluster_ard_variance(
            other, interaction_block, covariance[np.ix_(positions, positions)]
        )
        shrunk[positions] = _ard_interactions(
            other, interaction_block, root_weights * (y - y[0]), sigma_sq
        )
    return tuple(float(b) for b in shrunk)


def _validate_cate_inputs(y: np.ndarray, d: np.ndarray, alpha: float) -> None:
    if y.ndim != 1:
        _raise("estimation.cate.outcome_shape", shape=y.shape)
    if d.shape != y.shape:
        _raise("estimation.cate.treatment_shape_expected", d_shape=d.shape, y_shape=y.shape)
    if not np.isfinite(y).all():
        refuse(CATE_OUTCOME_NULLS_NON)

    if not (np.isfinite(d).all() and np.isin(d, (0.0, 1.0)).all()):
        _raise(
            "estimation.cate.treatment_binary",
            values=sorted(np.unique(d[~np.isin(d, (0.0, 1.0))]).tolist())[:5],
        )
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)


def fit_cate(  # noqa: PLR0915
    y: np.ndarray,
    d: np.ndarray,
    cols: Mapping[str, np.ndarray],
    *,
    interact: Sequence[Covariate],
    adjust: Sequence[Covariate] = (),
    alpha: float = 0.05,
    ard: bool = False,
    cluster_ids: np.ndarray | None = None,
    cluster_weight: Literal["member_count", "equal"] = "member_count",
    intervention_grain: Literal["unit", "cluster"] = "unit",
) -> CateResult:
    """Fit Lin's interacted regression of *y* on treatment *d*.

    The design is ``[1, d, M, d * Bc]``: ``M`` is every covariate's
    basis centered at its target-weighted mean, ``Bc`` the ``interact``
    basis centered at that same target - this centering makes the treatment
    coefficient the ATE rather than the effect at the origin.

    Without clusters, inference uses HC2 and rank pruning as before.
    With ``cluster_ids``, inference uses grouped weighted scores, scaled by
    K/(K-1), with t(K-1) intervals. All observed clusters count, including
    singletons and clusters with constant residuals; no HC2 divisor is used.
    ``cluster_weight="member_count"`` gives every row weight one; ``"equal"``
    gives weight 1/cluster_size and requires IDs. These weights determine
    basis centering, fitting, sandwich scores and residualized ARD inputs.
    ``intervention_grain`` never selects weights implicitly.

    Cluster rank deficiency, unsupported single-cluster directions, invalid
    covariance, and unavailable required uncertainty raise coded
    ``InvalidRequestError``. A singular interaction covariance refuses the
    required Wald test even when individual scalar SEs could be available.

    *ard* opts the interaction block into automatic relevance
    determination (see ``_ard_interactions``), moving the interaction
    point estimates only - ``ate``/``se``/``heterogeneity`` stay the
    unpenalized fit's. ``cate`` and ``contrast`` return an interval-free
    ``Estimate`` because the unpenalized sandwich does not provide nominal
    coverage for the shrunken point; ``score`` returns shrunken predictions.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    _validate_cate_inputs(y, d, alpha)
    n = int(y.size)
    cluster_index = _cluster_index(cluster_ids, n, intervention_grain)
    _validate_cluster_weight(cluster_weight, clustered=cluster_index is not None)
    groups = None
    n_clusters = None
    weights = np.ones(n)
    if cluster_index is not None:
        groups, sizes = cluster_index
        n_clusters = int(sizes.size)
        if n_clusters < 2:
            _raise("estimation.cate.insufficient_clusters", count=n_clusters)
        if cluster_weight == "equal":
            weights = 1.0 / sizes[groups]
    n_treated = int((d == 1.0).sum())
    n_control = n - n_treated
    if n_treated < 2 or n_control < 2:
        _raise("estimation.cate.each_arm_needs", n_treated=n_treated, n_control=n_control)
    # Center the unclustered solve and ARD; restore only the reported intercept.
    outcome_anchor = float(y[0]) if groups is None else 0.0
    centered_y = y - outcome_anchor if groups is None else y
    unadjusted_vcov = None
    if groups is None:
        # Welch difference in means on the same rows, for se_reduction.
        se_unadjusted = math.sqrt(
            float(centered_y[d == 1.0].var(ddof=1)) / n_treated
            + float(centered_y[d == 0.0].var(ddof=1)) / n_control
        )
        if not se_unadjusted > 0.0:
            _raise("estimation.cate.outcome_constant_within")
    else:
        assert n_clusters is not None
        _, unadjusted_vcov, unadjusted_directions = _cluster_sandwich(
            np.column_stack([np.ones(n), d]), y, weights, groups, n_clusters
        )
        loading = np.array([0.0, 1.0])
        _refuse_cluster_direction(loading, unadjusted_directions, query="unadjusted ATE")
        se_unadjusted = _cluster_se(loading, unadjusted_vcov, query="unadjusted ATE")

    interact_covs = _dedup(interact)
    main_covs = _dedup((*adjust, *interact))
    main_spec = DesignSpec.fit(cols, main_covs)
    interact_spec = DesignSpec.fit(cols, interact_covs)
    main_raw = _basis(main_spec, cols, n)
    interact_raw = _basis(interact_spec, cols, n)

    width = main_raw.shape[1] + interact_raw.shape[1]
    if width * _MAX_WIDTH_DENOM > n:
        _raise("estimation.cate.design_too_wide", width=width, n=n)

    interact_means = (
        interact_raw.mean(axis=0)
        if groups is None
        else np.average(interact_raw, axis=0, weights=weights)
    )
    main_means = (
        main_raw.mean(axis=0) if groups is None else np.average(main_raw, axis=0, weights=weights)
    )
    z = np.column_stack(
        [
            np.ones(n),
            d,
            main_raw - main_means,
            d[:, None] * (interact_raw - interact_means),
        ]
    )
    main_names = main_spec.column_names()
    interact_names = interact_spec.column_names()
    names = ("intercept", "d", *main_names, *(f"{_INTERACT}{m}" for m in interact_names))

    directions: tuple[tuple[float, ...], ...] = ()
    if groups is None:
        beta, vcov, resid, zk, kept, pruned, kept_names = _fit_hc2(
            z, centered_y, d, names, main_raw, main_spec, interact_raw, interact_spec
        )
    else:
        assert n_clusters is not None
        kept = tuple(range(z.shape[1]))
        pruned = ()
        zk, kept_names = z, names
        beta, vcov, directions = _cluster_sandwich(zk, y, weights, groups, n_clusters)
        for j in (1, *range(2 + len(main_names), zk.shape[1])):
            _refuse_cluster_direction(np.eye(zk.shape[1])[j], directions, query=kept_names[j])

    level = 1.0 - alpha
    dof = max(n - zk.shape[1], 1) if n_clusters is None else n_clusters - 1
    t_crit = two_sided_critical_value(student_t_isf, alpha, dof, what="CATE interaction interval")
    ate = float(beta[1])
    se = (
        math.sqrt(float(vcov[1, 1]))
        if groups is None
        else _cluster_se(np.eye(zk.shape[1])[1], vcov, query="ATE")
    )

    if groups is not None:
        _check_cluster_interval(ate, t_crit * se, query="ATE")

    offset = 2 + len(main_names)
    positions = [j for j, original in enumerate(kept) if original >= offset]
    interaction_positions = tuple(
        (j, original - offset) for j, original in enumerate(kept) if original >= offset
    )
    cluster_wald = None
    if groups is not None and positions:
        cluster_wald = _cluster_wald(beta[positions], vcov[np.ix_(positions, positions)], dof)

    if ard and positions:
        beta_ard = _shrink_interaction_block(
            beta,
            zk,
            centered_y,
            positions,
            vcov,
            weights,
            residual_variance=(
                float(resid @ resid) / (n - zk.shape[1]) if groups is None else None
            ),
        )
    else:
        beta_ard = None
    if groups is None:
        # All fitting and shrinkage used the centered response.
        beta[0] = math.fsum((float(beta[0]), outcome_anchor))
    effects = []
    for j in positions:
        coef = float(beta[j])
        coef_se = (
            math.sqrt(float(vcov[j, j]))
            if groups is None
            else _cluster_se(np.eye(zk.shape[1])[j], vcov, query=kept_names[j])
        )
        if groups is not None:
            _check_cluster_interval(coef, t_crit * coef_se, query=kept_names[j])
        effects.append(
            InteractionEffect(
                name=kept_names[j],
                coef=coef,
                se=coef_se,
                lb=coef - t_crit * coef_se,
                ub=coef + t_crit * coef_se,
                ard_coef=None if beta_ard is None else beta_ard[j],
            )
        )

    if positions:
        if cluster_wald is not None:
            heterogeneity = cluster_wald
        else:
            index = np.asarray(positions)
            gamma = beta[index]
            statistic = float(gamma @ np.linalg.solve(vcov[np.ix_(index, index)], gamma))
            heterogeneity = WaldTest(
                statistic=statistic,
                df=len(positions),
                p_value=float(chi2.sf(statistic, len(positions))),
            )
    else:
        heterogeneity = WaldTest(statistic=0.0, df=0, p_value=1.0)

    return CateResult(
        ate=ate,
        se=se,
        lb=ate - t_crit * se,
        ub=ate + t_crit * se,
        level=level,
        alpha=alpha,
        n=n,
        n_treated=n_treated,
        n_control=n_control,
        n_clusters=n_clusters,
        dimension=zk.shape[1],
        reference_df=dof,
        cluster_weight=cluster_weight,
        unadjusted_vcov=(
            None
            if unadjusted_vcov is None
            else tuple(tuple(float(v) for v in row) for row in unadjusted_vcov)
        ),
        unsupported_directions=directions,
        intervention_grain=intervention_grain,
        welch_se_unadjusted=se_unadjusted if n_clusters is None else None,
        heterogeneity=heterogeneity,
        interactions=tuple(effects),
        pruned=pruned,
        beta=tuple(float(b) for b in beta),
        vcov=tuple(tuple(float(v) for v in row) for row in vcov),
        columns=kept_names,
        main_spec=main_spec,
        interaction_spec=interact_spec,
        interaction_means=tuple(float(m) for m in interact_means),
        interaction_positions=interaction_positions,
        beta_ard=beta_ard,
    )
