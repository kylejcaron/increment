"""Cluster-score inference for fixed-horizon longitudinal change contrasts.

Each unit contributes weighted post minus uniform pre outcomes. Saturated
cohort/arm regressions standardize effects to the assigned roster; centered
pre-treatment covariates have separate slopes in each cohort/arm. Per-grain
Gram/XY summaries retain the cluster scores required for the sandwich.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.stats import t

from increment.errors import CodedModel, refuse
from increment.semantics.longitudinal import (
    LONGITUDINAL_ESTIMABILITY,
    LONGITUDINAL_NUMERIC,
    LongitudinalDesign,
    invalid,
)


class LongitudinalCluster(CodedModel, BaseModel):
    """One independent grain's additive regression sufficient statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    identity: str
    arm: str
    n_units: int
    gram: tuple[tuple[float, ...], ...]
    cross: tuple[float, ...]


class LongitudinalSummary(CodedModel, BaseModel):
    """Detached, owned cluster evidence; only trusted producers establish provenance.

    Outcome origins are removed per cohort before products are formed. Columns
    use cohort/arm blocks, each with intercept then cohort-centered covariates.
    Cohort counts are assigned-roster counts, never observed-row counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    design: LongitudinalDesign
    columns: tuple[str, ...]
    cohort_counts: tuple[int, ...]
    n_observations: int
    covariates: tuple[str, ...] = ()
    response_origins: tuple[float, ...]
    clusters: tuple[LongitudinalCluster, ...]

    @model_validator(mode="after")
    def _validate_summary(self) -> LongitudinalSummary:
        k = len(self.columns)
        if (
            len(self.response_origins) != 2 * len(self.design.cohorts)
            or not np.isfinite(self.response_origins).all()
        ):
            invalid("Summary requires finite response origins per cohort/arm.")
        if (
            not k
            or len(set(self.columns)) != k
            or k != 2 * len(self.design.cohorts) * (1 + len(self.covariates))
        ):
            invalid("Summary columns do not match the declared cohort/arm/covariate blocks.")
        if len(self.cohort_counts) != len(self.design.cohorts) or any(
            n <= 0 for n in self.cohort_counts
        ):
            invalid("Summary requires positive assigned-roster counts for every cohort.")
        if self.n_observations != sum(
            n * (len(c.pre_periods) + len(c.post_periods))
            for n, c in zip(self.cohort_counts, self.design.cohorts, strict=True)
        ):
            invalid("Summary observation count does not match the complete scheduled roster.")
        if len({g.identity for g in self.clusters}) != len(self.clusters):
            invalid("Summary independence identities must be unique.")
        stride = 1 + len(self.covariates)
        counted = [0.0] * len(self.design.cohorts)
        aggregate = np.zeros((k, k))
        for g in self.clusters:
            a = np.asarray(g.gram)
            if (
                g.arm not in (self.design.control_group, self.design.treatment_group)
                or g.n_units <= 0
            ):
                invalid("Summary has an invalid arm or unit count.")
            if (
                a.shape != (k, k)
                or len(g.cross) != k
                or not np.isfinite(a).all()
                or not np.isfinite(g.cross).all()
            ):
                invalid("Summary Gram/XY dimensions or finite values are invalid.")
            aggregate += a
            counts = [a[b * stride, b * stride] for b in range(2 * len(self.design.cohorts))]
            if any(n < 0 or n != int(n) for n in counts) or sum(counts) != g.n_units:
                invalid("Summary intercept counts must equal independent-grain unit counts.")
            arm_index = int(g.arm == self.design.treatment_group)
            for b, n in enumerate(counts):
                block_rows = slice(b * stride, (b + 1) * stride)
                if (b % 2 != arm_index or n == 0) and (
                    np.any(a[block_rows, :] != 0) or np.any(np.asarray(g.cross)[block_rows] != 0)
                ):
                    invalid(
                        "Summary block support conflicts with its independence-grain arm or count."
                    )
                for other in range(2 * len(self.design.cohorts)):
                    if b != other and np.any(
                        a[block_rows, other * stride : (other + 1) * stride] != 0
                    ):
                        invalid("Summary Gram matrices must retain disjoint cohort/arm blocks.")
                counted[b // 2] += n
            if not np.allclose(a, a.T, rtol=0, atol=1e-12):
                invalid("Summary Gram matrices must be symmetric.")
            if np.linalg.eigvalsh(a).min() < -1e-9 * max(1.0, float(np.max(np.abs(a)))):
                invalid("Summary Gram matrices must be positive semidefinite.")
        for j, n in enumerate(self.cohort_counts):
            control, treatment = 2 * j * stride, (2 * j + 1) * stride
            centered = (
                aggregate[control, control + 1 : control + stride]
                + aggregate[treatment, treatment + 1 : treatment + stride]
            )
            scale = np.sqrt(
                n
                * (
                    np.diag(aggregate)[control + 1 : control + stride]
                    + np.diag(aggregate)[treatment + 1 : treatment + stride]
                )
            )
            if np.any(np.abs(centered) > 1e-9 * np.maximum(1.0, scale)):
                invalid("Summary covariates must be centered over the full roster within cohort.")
        if tuple(counted) != self.cohort_counts:
            invalid("Summary cohort intercept counts must match the assigned roster.")
        if sum(g.n_units for g in self.clusters) != sum(self.cohort_counts):
            invalid("Summary independent-grain counts do not match the assigned roster.")
        if self.design.independence_grain == "unit" and any(g.n_units != 1 for g in self.clusters):
            invalid("A unit-randomized summary must retain one unit per independence grain.")
        return self


class LongitudinalResult(CodedModel, BaseModel):
    """One exploratory additive contrast with a working cluster-t reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    effect: float
    standard_error: float | None
    lower: float | None
    upper: float | None
    p_value: float | None
    unavailable_reason: Literal["zero_variance"] | None
    alpha: float
    degrees_of_freedom: int
    n_units: int
    n_independent: int
    control_independent: int
    treatment_independent: int
    n_observations: int
    design: LongitudinalDesign
    covariates: tuple[str, ...]
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    contrast: tuple[float, ...]
    cohort_effects: tuple[float, ...]
    guarantee: Literal["asymptotic"] = "asymptotic"
    reference: Literal["cluster_cr1_t"] = "cluster_cr1_t"
    multiplicity_status: Literal["exploratory_unadjusted"] = "exploratory_unadjusted"


def longitudinal_contrast(summary: LongitudinalSummary) -> np.ndarray:
    """Standardize cohort contrasts to equal weight per assigned roster unit."""
    stride = 1 + len(summary.covariates)
    l = np.zeros(len(summary.columns))
    for j, n in enumerate(summary.cohort_counts):
        l[2 * j * stride] = -n / sum(summary.cohort_counts)
        l[(2 * j + 1) * stride] = n / sum(summary.cohort_counts)
    return l


def _require_replicated_design(
    grams: np.ndarray, scaled_a: np.ndarray, scales: np.ndarray, stride: int
) -> None:
    # Removing one independence grain must leave every fitted direction identified.
    for gram in grams:
        for start in range(0, len(scales), stride):
            if gram[start, start] == 0:
                continue
            block = slice(start, start + stride)
            remaining = scaled_a[block, block] - gram[block, block] / np.outer(
                scales[block], scales[block]
            )
            eigenvalues = np.linalg.eigvalsh(remaining)
            if eigenvalues[0] <= 1e-12 * max(1.0, float(eigenvalues[-1])):
                refuse(
                    LONGITUDINAL_ESTIMABILITY,
                    message="A fitted covariate direction depends on one independent grain; obtain replicated levels or simplify the covariates.",
                )


def _finite_sum(values: Iterable[float]) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError):
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Longitudinal arithmetic exceeds floating-point resolution; rescale outcomes.",
        )
    if not math.isfinite(result):
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Longitudinal arithmetic exceeds floating-point resolution; rescale outcomes.",
        )
    return result


def _score_resolution(
    grams: np.ndarray, cross: np.ndarray, beta: np.ndarray, direction: np.ndarray
) -> float:
    """Conservative floating-point scale of cross-product and score cancellation."""
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude = np.abs(cross) + np.einsum("gij,j->gi", np.abs(grams), np.abs(beta))
        return float(32 * np.finfo(float).eps * len(beta) * (magnitude @ np.abs(direction)).sum())


def _unadjusted_projected_scores(summary: LongitudinalSummary) -> list[float]:
    # Near cancellation, resolve the scalar mean contrast exactly on retained doubles.
    k = len(summary.columns)
    counts = [sum(int(g.gram[i][i]) for g in summary.clusters) for i in range(k)]
    means = [
        sum((Fraction.from_float(g.cross[i]) for g in summary.clusters), Fraction()) / counts[i]
        for i in range(k)
    ]
    n = sum(summary.cohort_counts)
    direction = [
        Fraction((-1 if i % 2 == 0 else 1) * summary.cohort_counts[i // 2], n * counts[i])
        for i in range(k)
    ]
    result = []
    for g in summary.clusters:
        exact = sum(
            (
                direction[i] * (Fraction.from_float(g.cross[i]) - int(g.gram[i][i]) * means[i])
                for i in range(k)
            ),
            Fraction(),
        )
        value = float(exact)
        if exact and value == 0:
            refuse(
                LONGITUDINAL_NUMERIC,
                message="Contrast score is below floating-point resolution; rescale outcomes.",
            )
        result.append(value)
    return result


def fit_longitudinal(summary: LongitudinalSummary, *, alpha: float = 0.05) -> LongitudinalResult:
    """Fit the roster-standardized contrast using dependence-correct cluster scores.

    CR1 is G/(G-1), without a panel-row degrees-of-freedom multiplier. The
    Student-t reference with G-1 df is a working asymptotic approximation.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1 or alpha / 2 == 0:
        invalid("alpha must be finite and strictly between zero and one.", alpha=alpha)
    summary = LongitudinalSummary.model_validate(summary.model_dump())
    grams = np.asarray([g.gram for g in summary.clusters])
    cross = np.asarray([g.cross for g in summary.clusters])
    a = grams.sum(axis=0)
    stride = 1 + len(summary.covariates)
    for j in range(2 * len(summary.design.cohorts)):
        count = sum(g.gram[j * stride][j * stride] > 0 for g in summary.clusters)
        if count < 2:
            refuse(
                LONGITUDINAL_ESTIMABILITY,
                message="Each cohort/arm requires at least two independent grains.",
                block=j,
            )
    if np.linalg.matrix_rank(a, hermitian=True) < len(a):
        refuse(
            LONGITUDINAL_ESTIMABILITY,
            message="The covariate design is rank deficient; remove redundant or sparse predictors.",
        )
    scales = np.sqrt(np.diag(a))
    scaled_a = a / np.outer(scales, scales)
    if np.linalg.cond(scaled_a) > 1e12:
        refuse(
            LONGITUDINAL_NUMERIC,
            message="The covariate design is numerically ill conditioned; rescale or remove collinear predictors.",
        )
    _require_replicated_design(grams, scaled_a, scales, stride)
    beta = np.linalg.solve(scaled_a, cross.sum(axis=0) / scales) / scales
    bread = np.linalg.inv(scaled_a) / np.outer(scales, scales)
    scores = cross - np.einsum("gij,j->gi", grams, beta)
    g = len(grams)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = (bread @ (scores.T @ scores) @ bread) * (g / (g - 1))
    if not np.isfinite(covariance).all():
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Longitudinal covariance exceeds floating-point resolution; rescale outcomes.",
        )
    l = longitudinal_contrast(summary)
    reported_beta = beta.copy()
    reported_beta[::stride] += summary.response_origins
    cohort_effects = tuple(
        _finite_sum(
            (
                float(beta[(2 * j + 1) * stride]),
                -float(beta[2 * j * stride]),
                summary.response_origins[2 * j + 1] - summary.response_origins[2 * j],
            )
        )
        for j in range(len(summary.design.cohorts))
    )
    effect = _finite_sum(
        n / sum(summary.cohort_counts) * e
        for n, e in zip(summary.cohort_counts, cohort_effects, strict=True)
    )
    direction = bread @ l
    projected = [
        _finite_sum(float(x) * float(y) for x, y in zip(row, direction, strict=True))
        for row in scores
    ]
    resolution = _score_resolution(grams, cross, beta, direction)
    variance = _finite_sum(x * x for x in projected) * (g / (g - 1))
    if not summary.covariates and math.sqrt(variance) <= resolution:
        projected = _unadjusted_projected_scores(summary)
        variance = _finite_sum(x * x for x in projected) * (g / (g - 1))
    if summary.covariates and (
        not math.isfinite(resolution) or resolution > 1e-4 * math.sqrt(variance)
    ):
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Large fitted covariate effects leave unresolved residual precision; subtract a known treatment-unaffected outcome baseline before preparing the panel.",
        )
    if (
        not math.isfinite(effect)
        or not np.isfinite(covariance).all()
        or not math.isfinite(variance)
        or variance < 0
    ):
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Longitudinal fit exceeds numerical resolution; rescale outcomes/covariates.",
        )
    if variance == 0 and any(x != 0 for x in projected):
        refuse(
            LONGITUDINAL_NUMERIC,
            message="Contrast variance is below floating-point resolution; rescale outcomes.",
        )
    se = math.sqrt(variance) if variance > 0 else None
    critical = float(t.isf(alpha / 2, g - 1))
    lower = effect - critical * se if se is not None else None
    upper = effect + critical * se if se is not None else None
    if not math.isfinite(critical) or (
        lower is not None
        and (not math.isfinite(lower) or (upper is None or not math.isfinite(upper)))
    ):
        refuse(LONGITUDINAL_NUMERIC, message="The requested interval exceeds numerical resolution.")
    n_control = sum(c.arm == summary.design.control_group for c in summary.clusters)
    return LongitudinalResult(
        effect=effect,
        standard_error=se,
        lower=lower,
        upper=upper,
        p_value=float(2 * t.sf(abs(effect / se), g - 1)) if se is not None else None,
        unavailable_reason="zero_variance" if se is None else None,
        alpha=alpha,
        degrees_of_freedom=g - 1,
        n_units=sum(summary.cohort_counts),
        n_independent=g,
        control_independent=n_control,
        treatment_independent=g - n_control,
        n_observations=summary.n_observations,
        design=summary.design,
        covariates=summary.covariates,
        coefficients=tuple(reported_beta),
        covariance=tuple(tuple(row) for row in covariance),
        contrast=tuple(l),
        cohort_effects=cohort_effects,
    )
