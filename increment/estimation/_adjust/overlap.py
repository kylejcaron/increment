"""Identification and overlap helpers for observational adjustment."""

from __future__ import annotations

import math
import operator
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from increment._identity import canonical_id_strings
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation._adjust.encoding import (
    CovariateLayout,
    UnseenLevels,
    learner_name,
    modal_code,
    restricted,
)
from increment.estimation._adjust.learners import Learner
from increment.estimation.crossfit import (
    ESTIMATION_CROSSFIT_FOLD_ASSIGNMENTS_UNIT_IDS_SHAPE,
    _assigned_strata_counts,
    _balance_destinations,
    _grouped_strata,
    _pure_groups,
)


class IdentificationError(CodedError):
    """Identification gate refused: the data cannot support the requested estimand.

    Every call site names its own stable dotted code and carries structured
    context; there is no generic default.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        context: Mapping[str, object] = MappingProxyType({}),
    ) -> None:
        super().__init__(message, code=code, context=context)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.adjust_overlap.cross_fitting_needs": "cross-fitting needs at least 2 folds, got k={k}",
        "estimation.adjust_overlap.arms_shape": "arms must be 1-d, got shape {arms}",
        "estimation.adjust_overlap.arms_unit_ids": "arms and unit_ids must have the same length, got {arms} and {unit_ids}",
        "estimation.adjust_overlap.fold_ids_needs": "_fold_ids needs string-typed unit ids for a stable hash, got dtype {unit_ids!r} -- cast to str before calling (e.g. unit_frame's own unit_id column, which is already String-typed)",
        "estimation.adjust_overlap.arm_stratified_cross": "arm-stratified cross-fitting needs at least two arm labels, got {unique_arms}",
        "estimation.adjust_overlap.arm_unit_but": "arm {arm!r} has only {indices} unit(s), but cross-fitting with {k} folds needs at least {k} per arm",
        "estimation.adjust_overlap.cluster_ids_one": RefusalSpec(
            "estimation.adjust_overlap.cluster_ids_one",
            InvalidRequestError,
            lambda **_: "cluster_ids must be one-dimensional and align with unit_ids",
        ),
        "estimation.adjust_overlap.arm_cluster_but": "arm {arm!r} has only {clusters} cluster(s), but cross-fitting with {k} folds needs at least {k} per arm",
        "estimation.adjust_overlap.cluster_arm_needs_two": "treatment-pure cluster contrast needs at least 2 clusters in EACH arm to estimate a between-cluster variance for that arm, got {k_t} treated cluster(s) and {k_c} control cluster(s) -- a single cluster carries no between-cluster variance contribution to estimate its own arm's component from.",
        "estimation.adjust_overlap.cluster_needs_two": "cluster covariance needs at least two independent clusters, got {k}",
    },
)

_REFUSALS["estimation.crossfit.fold_assignments_unit_ids_shape"] = (
    ESTIMATION_CROSSFIT_FOLD_ASSIGNMENTS_UNIT_IDS_SHAPE
)
_raise = raiser(_REFUSALS)


def _fold_ids(
    unit_ids: np.ndarray,
    arms: np.ndarray,
    k: int,
    *,
    cluster_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Assign deterministic, arm-stratified cross-fit folds.

    Units - or, when cluster_ids is supplied, whole clusters - are
    canonicalized injectively (`_identity.canonical_id_strings`: a missing
    identity refuses, and two distinct native identities that stringify
    the same refuse rather than silently merge) and ranked by a stable
    content hash within each arm, then assigned round-robin. Each
    cluster's fold is broadcast to its units so no validation cluster can
    appear in training. An arm-pure cluster (every member shares one arm)
    round-robins exactly like the unclustered case. A mixed-treatment
    observational cluster (members span both arms) is never split to
    force arm purity - instead it is placed, in the same hash-ranked
    order, to balance the aggregate per-arm counts across folds (see
    `crossfit._balance_destinations`, the same primitive `outer_split`/
    `fold_assignments` use). If every cluster contains one unit,
    assignment falls back to ranking unit IDs so declaring a singleton
    cluster does not change the unclustered folds. Hash ranking (rather
    than input position) keeps membership invariant to warehouse row
    order while round-robin keeps each arm balanced across folds. Every
    arm needs at least k atomic groups touching it, combined across its
    *pure* clusters and every *mixed* cluster that also carries a row of
    it - not pure and mixed counts checked independently, since an arm
    thin on pure clusters can still be adequately covered by mixed
    clusters that carry it too, and refusing that as if the mixed
    clusters did not exist would reject a perfectly feasible split.

    unit_ids must be string-typed (str/unicode/bytes/object of str), rejected
    rather than silently hashing a numeric dtype's str() representation.
    arms must be a one-dimensional categorical arm label array aligned with
    unit_ids. Two labels reproduce the binary allocation; more labels use the
    same grouped allocator, whose mixed-cluster balancing is greedy, so
    callers still validate each fitted comparison's training support.
    """
    if k < 2:
        _raise("estimation.adjust_overlap.cross_fitting_needs", k=k)
    unit_ids = np.asarray(unit_ids)
    arms = np.asarray(arms)
    if unit_ids.ndim != 1:
        _raise("estimation.crossfit.fold_assignments_unit_ids_shape", unit_ids=unit_ids.shape)
    if arms.ndim != 1:
        _raise("estimation.adjust_overlap.arms_shape", arms=arms.shape)
    if arms.size != unit_ids.size:
        _raise("estimation.adjust_overlap.arms_unit_ids", arms=arms.size, unit_ids=unit_ids.size)
    if unit_ids.dtype.kind not in ("U", "S", "O"):
        _raise("estimation.adjust_overlap.fold_ids_needs", unit_ids=unit_ids.dtype)
    unique_arms = np.unique(arms)
    if unique_arms.size < 2:
        _raise("estimation.adjust_overlap.arm_stratified_cross", unique_arms=unique_arms.size)

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.ndim != 1 or cluster_ids.size != unit_ids.size:
            _raise("estimation.adjust_overlap.cluster_ids_one")

    groups, n_groups, keys, purity, mixed_rows, n_values = _grouped_strata(
        unit_ids, arms, cluster_ids
    )
    if cluster_ids is not None:
        units = canonical_id_strings(unit_ids, what="unit_ids")
        group_units = np.empty(n_groups, dtype=object)
        group_units[groups] = units
        if np.array_equal(group_units[groups], units) and np.unique(group_units).size == n_groups:
            keys = group_units
            cluster_ids = None
    key_strings = np.asarray(keys, dtype=str)
    hashes = np.array([zlib.crc32(key.encode()) for key in key_strings], dtype=np.uint32)

    mixed_ids = np.flatnonzero(purity == -1)
    touching = np.bincount(purity[purity >= 0], minlength=n_values).astype(np.int64)
    for g in mixed_ids.tolist():
        for v in mixed_rows[g]:
            touching[v] += 1
    for vi, arm in enumerate(unique_arms):
        if touching[vi] < k:
            code = (
                "estimation.adjust_overlap.arm_unit_but"
                if cluster_ids is None
                else "estimation.adjust_overlap.arm_cluster_but"
            )
            count_kwarg = "indices" if cluster_ids is None else "clusters"
            _raise(code, arm=arm, **{count_kwarg: int(touching[vi])}, k=k)

    dest_by_group = np.empty(n_groups, dtype=np.int64)
    for _value, pure_ids in _pure_groups(purity, n_values):
        ranked = pure_ids[np.lexsort((key_strings[pure_ids], hashes[pure_ids]))]
        dest_by_group[ranked] = np.arange(ranked.size, dtype=np.int64) % k
    if mixed_ids.size:
        ranked = mixed_ids[np.lexsort((key_strings[mixed_ids], hashes[mixed_ids]))]
        weights = np.full(k, 1.0 / k)
        initial = _assigned_strata_counts(groups, purity, dest_by_group, k, n_values)
        dest_by_group[ranked] = _balance_destinations(ranked, mixed_rows, weights, initial)
    return dest_by_group[groups]


def _cluster_index(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """(inverse index, K): each unit's cluster ordinal and the cluster count."""
    uniques, inv = np.unique(labels, return_inverse=True)
    return inv, int(uniques.size)


def _restrict_cluster_index(inv: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Densely re-index a subset of rows' cluster ordinals.

    ``inv`` holds the subset's ordinals among ``k`` sorted clusters. The
    clusters the subset occupies keep their relative order, so this equals
    `_cluster_index` of the subset's own labels without sorting them again.
    """
    present = np.zeros(k, dtype=bool)
    present[inv] = True
    rank = np.cumsum(present) - 1
    return rank[inv], int(np.count_nonzero(present))


def _dyadic_integers(values: np.ndarray) -> tuple[list[int], Fraction]:
    """Integers and one power of two ``unit`` with ``values == integers * unit``.

    A finite double is a 53-bit integer times a power of two, so shifting
    every significand to the smallest exponent present keeps sums and
    products of ``values`` exact in integer arithmetic.
    """
    finite = np.isfinite(values)
    if not finite.all():
        # No exact value exists: raise as converting the first such value does.
        float(values[np.argmin(finite)]).as_integer_ratio()
    mantissa, exponent = np.frexp(values)
    significand = np.ldexp(mantissa, 53).astype(np.int64)
    power = exponent.astype(np.int64) - 53
    nonzero = significand != 0
    if not nonzero.any():
        return [0] * len(significand), Fraction(1)
    base = int(power[nonzero].min())
    shift = np.where(nonzero, power - base, 0)
    unit = Fraction(1 << base) if base >= 0 else Fraction(1, 1 << -base)
    return list(map(operator.lshift, significand.tolist(), shift.tolist())), unit


def _cluster_sq(psi: np.ndarray, inv: np.ndarray, k: int) -> tuple[float, float]:
    """Stable centered complete-cluster totals and their scale.

    With ``psi == integers * unit`` (`_dyadic_integers`), cluster ``g``'s
    centered total is exactly ``(n * sum_g - count_g * sum) * unit / n``.
    """
    values = np.asarray(psi, dtype=float)
    n = len(values)
    counts = np.bincount(inv, minlength=k)
    integers, unit = _dyadic_integers(values)
    sums = [0] * k
    for value, group in zip(integers, inv.tolist(), strict=True):
        sums[group] += value
    whole = sum(sums)
    step = unit / n
    centered = [
        n * total - count * whole for total, count in zip(sums, counts.tolist(), strict=True)
    ]
    peak = max(map(abs, centered), default=0) * step
    if not peak:
        return 0.0, 1.0
    maximum = Fraction(float(np.finfo(float).max))
    square = peak * peak
    corrected_bound = square * k * k / max(k - 1, 1)
    if corrected_bound <= maximum and square >= float(np.finfo(float).tiny):
        scale = 1.0
    else:
        scale = max(float(min(peak, maximum)), math.ulp(0.0))
    # Integer true division rounds each exact total / scale correctly.
    ratio = step / Fraction(scale)
    normalized = math.fsum((total * ratio.numerator / ratio.denominator) ** 2 for total in centered)
    return float(normalized), float(scale)


def _cluster_purity(inv: np.ndarray, k: int, d: np.ndarray) -> np.ndarray:
    """Per-cluster arm label: ``1`` (every member treated), ``0`` (every
    member control), or ``-1`` (mixed: members span both arms -- only
    reachable for supported observational ingress; randomized/encouragement
    cluster ingress still enforces purity upstream)."""
    sum_d = np.bincount(inv, weights=d, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    purity = np.full(k, -1, dtype=np.int64)
    purity[sum_d == counts] = 1
    purity[sum_d == 0.0] = 0
    return purity


def _validate_pair_cluster_support(inv: np.ndarray, k: int, d: np.ndarray) -> None:
    """Require the support one comparison's covariance needs: two distinct
    clusters, and two pure clusters per arm when every cluster is arm-pure.

    ``inv``, ``k`` and ``d`` cover only the actual treatment/control rows of
    one comparison, indexed over the ``k`` clusters those rows occupy, so a
    cluster holding only other treatment arms never counts toward either
    requirement.
    """
    if k < 2:
        _raise("estimation.adjust_overlap.cluster_needs_two", k=k)
    purity = _cluster_purity(inv, k, d)
    if bool(np.all(purity >= 0)):
        k_t, k_c = int((purity == 1).sum()), int((purity == 0).sum())
        if k_t < 2 or k_c < 2:
            _raise("estimation.adjust_overlap.cluster_arm_needs_two", k_t=k_t, k_c=k_c)


def _cluster_reduction(psi: np.ndarray, inv: np.ndarray, k: int) -> tuple[float, float]:
    """Normalized superpopulation covariance numerator and its score scale.

    For a member-weighted target, U_g = sum_i psi_i includes the cluster's
    size times the estimated target. Independence is across sampled clusters;
    treatment purity does not authorize conditioning on random arm composition.
    Sum the complete U_g before squaring to retain cross-arm covariance.

    K/(K-1) is a Bessel convention, not a fitted-residual leverage correction.
    Neither these totals nor treatment labels identify a finite-sample Student
    law. Callers therefore use an asymptotic Normal reference, with small-K
    calibration still required for the actual nuisance fits and sampling law.
    Arm support is validated per comparison before scoring
    (`_validate_pair_cluster_support`), not inferred from these totals.
    """
    if k < 2:
        _raise("estimation.adjust_overlap.cluster_needs_two", k=k)
    variance, scale = _cluster_sq(psi, inv, k)
    return variance * (k / (k - 1)), scale


def _moment_dicts(
    summary: IntoDataFrame | Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize a ``MomentSource.moments()`` result to row mappings."""
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        return list(frame.iter_rows(named=True))
    rows = cast("Iterable[Mapping[str, Any]]", summary)
    return [dict(r) for r in rows]


def _clustered_fields(
    psi: np.ndarray, inv: np.ndarray | None, k: int | None
) -> tuple[dict[str, Any], float | None]:
    """Cluster sandwich fields; scalar scores supply no finite-sample df."""
    if inv is None or k is None:
        return {}, None
    variance, scale = _cluster_reduction(psi, inv, k)
    return {
        "cluster_variance": variance,
        "n_clusters": k,
        "cluster_score_scale": scale,
    }, None


@dataclass(frozen=True, slots=True)
class SmdArms:
    """One comparison's treated and control rows and their weights, shared by
    every covariate's `_weighted_smd`."""

    treated: np.ndarray
    control: np.ndarray
    w_treated: np.ndarray
    w_control: np.ndarray
    total_treated: float
    total_control: float
    n_treated: int
    n_control: int

    @classmethod
    def of(cls, treated: np.ndarray, control: np.ndarray, w: np.ndarray) -> SmdArms:
        """Arms from boolean row masks and the ATE weights on every row."""
        w_treated, w_control = w[treated], w[control]
        return cls(
            treated,
            control,
            w_treated,
            w_control,
            float(w_treated.sum()),
            float(w_control.sum()),
            int(treated.sum()),
            int(control.sum()),
        )


def _weighted_smd(x: np.ndarray, arms: SmdArms) -> float:
    """Standardized mean difference of covariate *x* between arms: the
    weighted mean difference under the ATE weights `w` that `arms` carries,
    over a FIXED unweighted pooled sd `sqrt((s1**2 + s0**2) / 2)` from each
    arm's own (unweighted) sample variance -- the Austin & Stuart (2015)
    convention.

    The denominator deliberately does not depend on `w`: a weighted pooled
    sd is a function of the weights this gate exists to diagnose, and an
    inflating weight scheme (a few units near the overlap boundary
    dominating the weighted variance) shrinks a weighted denominator
    enough to slip an imbalanced covariate past `gate.max_smd` in exactly
    the extreme-weight regime the gate is meant to catch.

    A zero pooled sd carries no scale, so the standardized difference is
    undefined rather than zero. Arms sitting at the same level are genuinely
    balanced and return 0.0; arms at different levels with no within-arm
    spread are complete separation -- the worst imbalance there is -- and
    return +inf so a configured `gate.max_smd` refuses instead of reading the
    worst case as the best.

    That comparison is made on the raw covariate levels, not on the weighted
    means. The means are only equal up to the rounding of their weighted sums,
    so comparing them needs a tolerance, and any relative tolerance equates
    distinct levels that are close relative to their magnitude -- 1e9 and
    1e9 + 0.5 differ by less than 1e-9 relatively, and would have slipped
    complete separation past the gate again. The raw levels are exact.
    """
    x_treated, x_control = x[arms.treated], x[arms.control]
    m1 = float((arms.w_treated * x_treated).sum() / arms.total_treated)
    m0 = float((arms.w_control * x_control).sum() / arms.total_control)
    v1 = float(x_treated.var(ddof=1)) if arms.n_treated > 1 else 0.0
    v0 = float(x_control.var(ddof=1)) if arms.n_control > 1 else 0.0
    pooled_sd = np.sqrt((v1 + v0) / 2)
    if pooled_sd == 0:
        levels_t, levels_c = np.unique(x_treated), np.unique(x_control)
        return 0.0 if np.array_equal(levels_t, levels_c) else math.inf
    return float((m1 - m0) / pooled_sd)


def _contrast_text(treatment_groups: tuple[str, ...], control_group: str) -> str:
    """Message label for one comparison or for every comparison sharing a cohort."""
    if len(treatment_groups) == 1:
        return f"contrast {treatment_groups[0]!r} vs {control_group!r}"
    named = ", ".join(repr(group) for group in treatment_groups)
    return f"contrasts {named} vs {control_group!r}"


def _propensity_ranges(
    marginal: np.ndarray, groups: tuple[str, ...]
) -> tuple[tuple[str, float, float], ...]:
    """Per-arm (group, min, max) of determined marginal propensities.

    A unit whose conditional fits saturate for two treatments has an
    undetermined marginal split among them (NaN); it already fails the gate
    through its zero control propensity and is excluded from the ranges.
    """
    ranges = []
    for code, group in enumerate(groups):
        column = marginal[:, code]
        finite = column[np.isfinite(column)]
        if finite.size:
            ranges.append((group, float(finite.min()), float(finite.max())))
        else:
            ranges.append((group, math.nan, math.nan))
    return tuple(ranges)


def _infeasible_band_note(n_arms: int, g: float) -> str:
    """Explain why a requested common support band cannot hold for any unit."""
    if n_arms * g <= 1.0:
        return ""
    return (
        f"; with {n_arms} arms the marginal propensities sum to one, so every unit has "
        f"some arm propensity at most 1/{n_arms} < gate.min_propensity={g:g} and no unit "
        "can meet this common band"
    )


def _overlap_refusal_message(
    arm: np.ndarray,
    marginal: np.ndarray,
    outside: np.ndarray,
    g: float,
    groups: tuple[str, ...],
    *,
    method: str,
) -> str:
    """Name the arms and propensities that failed the common overlap gate.

    ``groups`` lists group labels by arm code, control first; column ``a`` of
    ``marginal`` is arm ``a``'s marginal propensity.
    """
    suffix = (
        " -- set gate.overlap='trim' to analyse the overlap subpopulation \u2014 trimming "
        "CHANGES the estimand and is recorded on LiftEstimate.population"
    )
    control_group, treatment_groups = groups[0], groups[1:]
    if len(treatment_groups) == 1:
        e = marginal[:, 1]
        treated = arm == 1
        n_treat, n_ctrl = int(treated.sum()), int((~treated).sum())
        out_treat = int((outside & treated).sum())
        out_ctrl = int((outside & ~treated).sum())
        deciles = ", ".join(f"{q:.3f}" for q in np.percentile(e, np.arange(0, 101, 10)))
        return (
            f"{method} overlap gate refused for "
            f"{_contrast_text(treatment_groups, control_group)}: "
            f"{out_treat} of {n_treat} treatment units and {out_ctrl} of {n_ctrl} control units "
            f"have fitted propensity outside [{g:g}, {1 - g:g}] "
            f"(e min={e.min():.4f}, max={e.max():.4f}; deciles [{deciles}])" + suffix
        )
    counts = ", ".join(
        f"{int((outside & (arm == code)).sum())} of {int((arm == code).sum())} in {group!r}"
        for code, group in enumerate(groups)
    )
    ranges = ", ".join(
        f"{group!r} [{low:.4f}, {high:.4f}]"
        for group, low, high in _propensity_ranges(marginal, groups)
    )
    return (
        f"{method} overlap gate refused for {_contrast_text(treatment_groups, control_group)}: "
        f"units with some marginal arm propensity below {g:g}: {counts} "
        f"(marginal propensity ranges {ranges}){_infeasible_band_note(len(groups), g)}" + suffix
    )


_MAX_MISSINGNESS_PATTERNS = 20
_MIN_PATTERN_N = 30


def _missing_covariate_refusal(
    miss: np.ndarray,
    covariates: list[str],
    *,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> IdentificationError:
    n = miss.shape[0]
    missing_covariates = tuple(
        (name, int(miss[:, j].sum())) for j, name in enumerate(covariates) if miss[:, j].any()
    )
    named = ", ".join(f"{name} ({count} of {n} missing)" for name, count in missing_covariates)
    return IdentificationError(
        f"{method}: adjustment covariate(s) with missing (null/NaN/non-finite) "
        f"values for metric {metric_name!r} "
        f"({_contrast_text(treatment_groups, control_group)}): {named}. NaN compares "
        f"False against every gate "
        f"threshold, so the overlap and balance gates cannot protect this "
        f"comparison. Keep every unit by declaring "
        f"AdjustmentSet(missing='impute-indicator') (pooled-mean impute plus a "
        f"missingness indicator appended to the adjustment set), "
        f"AdjustmentSet(missing='pattern') (propensity fit separately per "
        f"missingness pattern), or AdjustmentSet(missing='allow') (NaN passed "
        f"through raw -- requires explicitly supplied NaN-native learner(s) via "
        f"Method(propensity_learner=..., outcome_learner=...); the package "
        f"defaults silently emit non-finite predictions under NaN), or repair "
        f"the columns upstream while keeping a missingness indicator (mean "
        f"imputation alone can hide residual confounding).",
        code="adjust.identification.missing_covariates",
        context={
            "method": method,
            "metric_name": metric_name,
            "treatment_groups": treatment_groups,
            "control_group": control_group,
            "missing_covariates": missing_covariates,
            "n": n,
        },
    )


def _impute_with_indicators(
    X: np.ndarray,
    layout: CovariateLayout,
    miss: np.ndarray,
    *,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> tuple[np.ndarray, CovariateLayout, str]:
    """Pooled-mean impute each incomplete covariate and append its
    missingness indicator as a covariate in its own right.

    The pooled mean is computed over every compared arm jointly, never per
    arm, and the indicator enters the adjustment set, so it is balanced by the
    weights and shows up in the SMD diagnostics instead of hiding
    residual confounding behind a bare imputation. A categorical covariate
    takes its pooled modal level, the analogue of the pooled mean, and the
    note says so per column kind. Returns `(X_extended, layout_extended,
    note)`; a complete covariate adds no indicator, a fully-missing one
    cannot be imputed and is refused.
    """
    X = X.copy()
    names: list[str] = []
    indicators: list[np.ndarray] = []
    # (covariate, missing count, "mean" | "mode") per imputed column.
    imputed: list[tuple[str, int, str]] = []
    n = X.shape[0]
    for j, name in enumerate(layout.names):
        col_miss = miss[:, j]
        if not col_miss.any():
            continue
        if col_miss.all():
            raise IdentificationError(
                f"{method}: adjustment covariate {name!r} has no observed "
                f"values for metric {metric_name!r} "
                f"({_contrast_text(treatment_groups, control_group)}) -- every row is "
                f"missing, so there is nothing to impute a pooled mean from. "
                f"Drop it from the adjustment set or repair the column "
                f"upstream.",
                code="adjust.identification.covariate_fully_missing",
                context={
                    "method": method,
                    "metric_name": metric_name,
                    "treatment_groups": treatment_groups,
                    "control_group": control_group,
                    "covariate": name,
                },
            )
        if layout.levels[j] is None:
            X[col_miss, j] = X[~col_miss, j].mean()
            imputed.append((name, int(col_miss.sum()), "mean"))
        else:
            X[col_miss, j] = modal_code(X[:, j], ~col_miss)
            imputed.append((name, int(col_miss.sum()), "mode"))
        names.append(f"{name}__missing")
        indicators.append(col_miss.astype(float))
    X_ext = np.column_stack([X, *indicators])
    kinds = {kind for _, _, kind in imputed}
    if len(kinds) == 1:
        how = f"pooled-{kinds.pop()}"
        parts = [f"{name} ({count} of {n})" for name, count, _ in imputed]
    else:
        how = "pooled-mean (numeric) or pooled-mode (categorical)"
        parts = [f"{name} ({count} of {n}, {kind})" for name, count, kind in imputed]
    note = (
        f"missing covariate values {how} imputed with missingness "
        f"indicator(s) appended to the adjustment set: {', '.join(parts)}"
    )
    return X_ext, layout.extended(names), note


def _pattern_propensities(
    X: np.ndarray,
    d: np.ndarray,
    pair: slice | np.ndarray,
    miss: np.ndarray,
    learner: Learner,
    layout: CovariateLayout,
    *,
    unseen: UnseenLevels,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> tuple[np.ndarray, int]:
    """Fit the propensity separately per missingness pattern: the
    generalized propensity e(X_observed, R), which balances the observed
    covariates and the pattern itself with no assumption on the
    missing-data mechanism.

    ``d`` indicates the compared treatment over every cohort row and
    ``pair`` selects that treatment/control comparison's rows. Each
    pattern's model trains on the comparison's rows with that pattern's
    observed columns only (a categorical column's levels encoded within
    that fit), then predicts the conditional propensity for every cohort
    row sharing the pattern; a pattern with no observed columns
    gets its comparison arm share as an intercept-only propensity. The
    pattern-count cap and per-pattern floor (counted on the comparison's
    rows) refuse the scattered-missingness regime, where patterns go thin
    and every per-pattern fit is noise. Levels a pattern fit never saw in
    its prediction rows are recorded on ``unseen``. Returns
    `(e, n_patterns)`; the overlap and balance gates then run on the
    pooled `e` exactly as for a single fit.
    """
    patterns = np.unique(miss, axis=0)
    n_patterns = patterns.shape[0]
    contrast = _contrast_text(treatment_groups, control_group)
    if n_patterns > _MAX_MISSINGNESS_PATTERNS:
        raise IdentificationError(
            f"{method}: missing='pattern' found {n_patterns} distinct "
            f"missingness patterns for metric {metric_name!r} ({contrast}), above the "
            f"{_MAX_MISSINGNESS_PATTERNS}-pattern cap -- scattered "
            f"per-covariate missingness makes every per-pattern propensity "
            f"fit too thin to trust. Declare missing='impute-indicator' "
            f"instead, or coarsen the missingness upstream (impute the "
            f"rarely-missing covariates so only the structural patterns "
            f"remain).",
            code="adjust.identification.pattern_cap_exceeded",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "n_patterns": n_patterns,
                "cap": _MAX_MISSINGNESS_PATTERNS,
            },
        )
    e = np.empty(d.shape[0], dtype=float)
    fit_mask = None if isinstance(pair, slice) else pair
    for pattern in patterns:
        rows = (miss == pattern).all(axis=1)
        train = rows if fit_mask is None else rows & fit_mask
        n_pattern = int(train.sum())
        if n_pattern < _MIN_PATTERN_N:
            missing_names = [name for j, name in enumerate(layout.names) if pattern[j]]
            label = f"missing {missing_names!r}" if missing_names else "fully observed"
            raise IdentificationError(
                f"{method}: missing='pattern' has a missingness pattern "
                f"({label}) with only {n_pattern} unit(s) for metric "
                f"{metric_name!r} ({contrast}), below the {_MIN_PATTERN_N}-unit floor "
                f"for a per-pattern propensity fit. Declare "
                f"missing='impute-indicator' instead, or coarsen the "
                f"missingness upstream (impute the rarely-missing covariates "
                f"so only the structural patterns remain).",
                code="adjust.identification.pattern_below_floor",
                context={
                    "method": method,
                    "metric_name": metric_name,
                    "treatment_groups": treatment_groups,
                    "control_group": control_group,
                    "missing_covariates": tuple(missing_names),
                    "n_pattern": n_pattern,
                    "floor": _MIN_PATTERN_N,
                },
            )
        observed = ~pattern
        if not observed.any():
            # No covariate observed for this pattern: the honest propensity
            # is intercept-only (the pattern's arm share); the overlap gate still fires on it.
            e[rows] = float(d[train].mean())
        else:
            from increment.estimation._adjust.common import _prediction_shape_guard

            model = restricted(learner, observed)
            model.fit(X[np.ix_(train, observed)], d[train])
            X_rows = X[np.ix_(rows, observed)]
            pred = model.predict(X_rows)
            unseen.record(model, X_rows, rows)
            e[rows] = _prediction_shape_guard(
                pred,
                expected_n=int(rows.sum()),
                learner_name=learner_name(learner),
                what="propensity",
                metric_name=metric_name,
                treatment_groups=treatment_groups,
                control_group=control_group,
                method=method,
            )
    return e, n_patterns


# missing="allow": NaN routed raw to explicitly supplied NaN-native learner(s);
# gates run on the imputed+indicator representation, guarded by 3 refusal layers.

_PROBE_ROWS = 64


def _allow_requires_learner_refusal(
    *, method: str, metric_name: str, defaulted: list[str]
) -> IdentificationError:
    """Config-level refusal for missing='allow' with defaulted learner(s).

    Unconditional (fires with or without NaN in today's data): the
    defaults are measured-incapable, since LogisticPropensity and
    RidgeOutcome do not raise on NaN input, they silently return
    non-finite predictions, and NaN compares False against every gate
    threshold.
    """
    slots = " and ".join(f"Method({name}=...)" for name in defaulted)
    return IdentificationError(
        f"{method}: AdjustmentSet(missing='allow') passes NaN straight to the "
        f"learner(s), so every learner that consumes X must be explicitly "
        f"supplied (metric {metric_name!r}); defaulted: {', '.join(defaulted)}. "
        f"The package defaults do not raise on NaN input -- they silently "
        f"return non-finite predictions, which the NaN-blind gates cannot "
        f"catch. Supply {slots} with NaN-native learner(s) (e.g. a LightGBM "
        f"or HistGradientBoosting wrapper). This refusal is unconditional: "
        f"declaring missing='allow' while running the defaults is a "
        f"configuration error whether or not the current data contains NaN.",
        code="adjust.identification.allow_requires_learner",
        context={"method": method, "metric_name": metric_name, "defaulted": tuple(defaulted)},
    )


def _allow_nan_mask(
    X: np.ndarray,
    covariates: list[str],
    *,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> np.ndarray:
    """The routable-missingness mask under missing='allow': NaN only.

    A NaN-native learner's split routing handles NaN, not +/-inf, so any
    non-finite non-NaN value refuses loudly by column name instead of
    riding along as if it were missing.
    """
    nan_mask = np.isnan(X)
    inf_mask = ~np.isfinite(X) & ~nan_mask
    if inf_mask.any():
        n = X.shape[0]
        nonfinite_covariates = tuple(
            (name, int(inf_mask[:, j].sum()))
            for j, name in enumerate(covariates)
            if inf_mask[:, j].any()
        )
        named = ", ".join(
            f"{name} ({count} of {n} non-finite)" for name, count in nonfinite_covariates
        )
        raise IdentificationError(
            f"{method}: missing='allow' passes NaN through to the supplied "
            f"NaN-native learner(s), but adjustment covariate(s) contain "
            f"non-finite non-NaN (+/-inf) values for metric {metric_name!r} "
            f"({_contrast_text(treatment_groups, control_group)}): {named}. "
            f"A NaN-native learner's split routing handles NaN only -- "
            f"repair the +/-inf values upstream.",
            code="adjust.identification.allow_nonfinite_covariates",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "nonfinite_covariates": nonfinite_covariates,
                "n": n,
            },
        )
    return nan_mask


def _allow_note(miss: np.ndarray, layout: CovariateLayout) -> str:
    """The estimate note under missing='allow', carrying the
    learner-capacity caveat: NaN tolerance alone is not capacity. The gate
    representation is named per column kind: a null numeric column was
    pooled-mean imputed there, a null categorical one pooled-mode imputed."""
    n = miss.shape[0]
    nulled = [j for j in range(len(layout.names)) if miss[:, j].any()]
    named = ", ".join(f"{layout.names[j]} ({int(miss[:, j].sum())} of {n} missing)" for j in nulled)
    kinds = {"mean" if layout.levels[j] is None else "mode" for j in nulled}
    representation = (
        f"pooled-{kinds.pop()}-imputed"
        if len(kinds) == 1
        else "pooled-mean (numeric) / pooled-mode (categorical) imputed"
    )
    return (
        f"NaN covariate values passed through to NaN-native learner(s): {named}; "
        "assumes ignorability given observed values and the missingness "
        "pattern, AND that the learner can represent pattern-specific "
        "(indicator x covariate) structure -- additive or surrogate-split NaN "
        f"handling does not deliver this; gates computed on the "
        f"{representation} + indicator representation plus "
        "indicator x covariate interaction SMDs"
    )


def _probe_nan_capability(
    learner: Learner,
    n_cols: int,
    nan_cols: np.ndarray,
    *,
    role: str,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> None:
    """Fit/predict *learner* on a small deterministic NaN-planted matrix of
    X's width before the real fits, refusing on raise OR non-finite
    predictions. The finiteness half is load-bearing: the measured
    default-learner failure mode is silent NaN, not an exception.
    """
    rng = np.random.default_rng(0)
    x_probe = rng.standard_normal((_PROBE_ROWS, n_cols))
    for j in np.flatnonzero(nan_cols):
        x_probe[j % 3 :: 3, j] = np.nan
    label = (np.arange(_PROBE_ROWS) % 2).astype(float)
    name = learner_name(learner)
    try:
        learner.fit(x_probe, label)
        pred = np.asarray(learner.predict(x_probe), dtype=float)
    except Exception as err:
        raise IdentificationError(
            f"{method}: missing='allow' {role} learner {name} failed the "
            f"NaN-capability probe for metric {metric_name!r} "
            f"({_contrast_text(treatment_groups, control_group)}) -- fit/predict raised "
            f"on a {_PROBE_ROWS}x{n_cols} matrix with NaN planted in the "
            f"column(s) missing in the data: {err}. Supply a NaN-native "
            f"learner (e.g. a LightGBM or HistGradientBoosting wrapper).",
            code="adjust.identification.allow_probe_raised",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "role": role,
                "learner_name": name,
                "error": str(err),
            },
        ) from err
    n_bad = int((~np.isfinite(pred)).sum())
    if n_bad:
        raise IdentificationError(
            f"{method}: missing='allow' {role} learner {name} failed the "
            f"NaN-capability probe for metric {metric_name!r} "
            f"({_contrast_text(treatment_groups, control_group)}) -- {n_bad} of "
            f"{pred.shape[0]} probe predictions are non-finite under NaN "
            f"input, and NaN compares False against every gate threshold, so "
            f"this would ship past the overlap and balance gates. Supply a "
            f"NaN-native learner (e.g. a LightGBM or HistGradientBoosting "
            f"wrapper).",
            code="adjust.identification.allow_probe_nonfinite",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "role": role,
                "learner_name": name,
                "n_bad": n_bad,
                "n_total": pred.shape[0],
            },
        )


def _allow_fit_predict(
    learner: Learner,
    X_train: np.ndarray,
    label: np.ndarray,
    X_test: np.ndarray,
    *,
    stage: str,
    what: str,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> np.ndarray:
    """One allow-path learner fit/predict, converting any raise into a
    named refusal (e.g. a covariate ALL-NaN within one fold's train-arm
    subset can legitimately make a NaN-native learner raise, which the
    global all-NaN refusal and the probe both miss).
    """
    try:
        learner.fit(X_train, label)
        pred = learner.predict(X_test)
    except Exception as err:
        raise IdentificationError(
            f"{method}: missing='allow' learner {learner_name(learner)} "
            f"raised during {stage} for metric {metric_name!r} "
            f"({_contrast_text(treatment_groups, control_group)}): {err}",
            code="adjust.identification.allow_fit_predict_raised",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "learner_name": learner_name(learner),
                "stage": stage,
                "error": str(err),
            },
        ) from err
    from increment.estimation._adjust.common import _prediction_shape_guard

    return _prediction_shape_guard(
        pred,
        expected_n=X_test.shape[0],
        learner_name=learner_name(learner),
        what=what,
        metric_name=metric_name,
        treatment_groups=treatment_groups,
        control_group=control_group,
        method=method,
    )


def _finite_guard(
    values: np.ndarray,
    *,
    what: str,
    learner_name: str,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
    allow_mode: bool,
) -> None:
    """Post-fit hard guard: a learner can emit NaN on a real region even
    when it never raises. Runs on every fitted e/e_hat/m_hat regardless of
    missing-data policy -- NaN compares False against every gate
    threshold, so an unguarded NaN nuisance silently disables the overlap
    and balance gates and ships an all-NaN estimate (adjust.py's own
    comment used to admit "the gates below are NaN-blind" here)."""
    bad = int((~np.isfinite(values)).sum())
    if not bad:
        return
    if allow_mode:
        raise IdentificationError(
            f"{method}: missing='allow' learner {learner_name} produced {bad} "
            f"of {values.shape[0]} non-finite {what} predictions for metric "
            f"{metric_name!r} ({_contrast_text(treatment_groups, control_group)}) "
            f"-- NaN compares False against every gate "
            f"threshold, so non-finite nuisances would silently disable the "
            f"overlap and balance gates. The learner passed the NaN probe but "
            f"emitted non-finite values on the real data; supply a learner "
            f"with full NaN support.",
            code="adjust.identification.nonfinite_nuisance",
            context={
                "method": method,
                "metric_name": metric_name,
                "treatment_groups": treatment_groups,
                "control_group": control_group,
                "learner_name": learner_name,
                "what": what,
                "n_bad": bad,
                "n_total": values.shape[0],
                "allow_mode": allow_mode,
            },
        )
    raise IdentificationError(
        f"{method}: learner {learner_name} produced {bad} of {values.shape[0]} "
        f"non-finite {what} predictions for metric {metric_name!r} "
        f"({_contrast_text(treatment_groups, control_group)}) -- NaN compares False "
        f"against every gate threshold, so non-finite nuisances would "
        f"silently disable the overlap and balance gates. Supply a learner "
        f"that produces finite predictions on this data.",
        code="adjust.identification.nonfinite_nuisance",
        context={
            "method": method,
            "metric_name": metric_name,
            "treatment_groups": treatment_groups,
            "control_group": control_group,
            "learner_name": learner_name,
            "what": what,
            "n_bad": bad,
            "n_total": values.shape[0],
            "allow_mode": allow_mode,
        },
    )


def _propensity_range_guard(
    e: np.ndarray,
    *,
    learner_name: str,
    metric_name: str,
    treatment_groups: tuple[str, ...],
    control_group: str,
    method: str,
) -> None:
    """Hard refusal for a fitted propensity outside [0, 1], distinct from
    the overlap gate: the overlap gate trims values INSIDE [0, 1] that sit
    close to the boundary (a legitimate positivity concern), while a value
    outside [0, 1] means the learner itself is broken. Under
    gate.overlap='trim' an out-of-range value would otherwise fall inside
    the trim band's comparison and get silently dropped as if it were a
    subpopulation, rather than reported as a broken model."""
    bad = (e < 0.0) | (e > 1.0)
    n_bad = int(bad.sum())
    if not n_bad:
        return
    bad_vals = e[bad]
    raise IdentificationError(
        f"{method}: learner {learner_name} produced {n_bad} propensity "
        f"prediction(s) outside [0, 1] (min={np.min(bad_vals):.3g}, "
        f"max={np.max(bad_vals):.3g}) for metric {metric_name!r} "
        f"({_contrast_text(treatment_groups, control_group)}) -- this is a broken "
        f"propensity model, not a positivity violation; the overlap gate "
        f"only trims values inside [0, 1] near the boundary. Supply a "
        f"propensity learner whose predictions are probabilities.",
        code="adjust.identification.propensity_out_of_range",
        context={
            "method": method,
            "metric_name": metric_name,
            "treatment_groups": treatment_groups,
            "control_group": control_group,
            "learner_name": learner_name,
            "n_bad": n_bad,
            "min_value": float(np.min(bad_vals)),
            "max_value": float(np.max(bad_vals)),
        },
    )


def _allow_interaction_smds(
    X_gate: np.ndarray,
    names: tuple[str, ...],
    sources: tuple[str, ...],
    base_covariates: list[str],
    arms: SmdArms,
) -> list[tuple[str, float]]:
    """Indicator x covariate interaction SMD rows, allow-only.

    Marginal SMDs are blind to pattern-specific confounding (an additive
    NaN-native learner can balance every marginal column while the
    within-pattern effect is inverted), so each {c}__missing indicator is
    crossed with every OTHER imputed base covariate -- each of a
    categorical's level columns in turn. Self-crosses are collinear with
    the indicator row (the imputed value is constant where the indicator
    is 1); indicator x indicator crosses are 'pattern''s enumeration job,
    not this diagnostic's. `names`/`sources` describe the gate design's
    columns: an indicator column is its own source, a level column's
    source is its covariate.
    """
    base = set(base_covariates)
    indicators = [
        (idx, name[: -len("__missing")])
        for idx, name in enumerate(names)
        if name.endswith("__missing") and name[: -len("__missing")] in base
    ]
    rows: list[tuple[str, float]] = []
    for idx, c in indicators:
        for b_idx, (b_name, b_source) in enumerate(zip(names, sources, strict=True)):
            if b_source == c or b_source not in base:
                continue
            rows.append(
                (f"{names[idx]}*{b_name}", _weighted_smd(X_gate[:, idx] * X_gate[:, b_idx], arms))
            )
    return rows
