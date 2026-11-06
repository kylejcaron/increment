"""Parameter-independent joint categorical moments and exact compact fitting."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from increment.estimation.armstats import ArmStats
from increment.estimation.cuped._numeric import _arm_mean, _arm_offset, _fail
from increment.estimation.cuped.encoding import _categorical_values, _encode_covariates
from increment.estimation.cuped.inference import _mean_covariance, finalize_fit
from increment.estimation.cuped.multicovariate import (
    _fit_calculator,
    _make_inference_summary,
    _validate_request,
)
from increment.estimation.cuped.projections import _cache_projections
from increment.estimation.cuped.types import (
    CategoricalCells,
    FitState,
    MulticovariateCupedFit,
    _freeze_payload,
    _invalid_evidence,
    _pattern_key,
)
from increment.estimation.variance import ratio_denominator_precision


def categorical_cells(
    y,
    arms,
    *,
    identity,
    control,
    arm_order,
    covariates,
    missing=None,
    numerator=None,
    denominator=None,
) -> CategoricalCells:
    """Reduce explicit observations to complete joint-pattern centered moments."""
    kinds = dict.fromkeys(covariates, "categorical")
    groups, order, indices, _, outcomes, covariates, _, missing = _validate_request(
        y,
        arms,
        control,
        arm_order,
        covariates,
        kinds,
        missing,
        numerator,
        denominator,
        require_arm_support=False,
    )
    columns = []
    for name, raw in covariates.items():
        values, null = _categorical_values(raw, np.asarray(raw), len(groups))
        if values.shape != groups.shape:
            _fail(
                "estimation.cuped_linear.input_shape", name=name, n=len(groups), shape=values.shape
            )
        policy = missing.get(name, "refuse")
        if policy not in ("refuse", "impute-indicator") or (null.any() and policy == "refuse"):
            _fail("estimation.cuped_linear.missing_refused", name=name)
        if any(not isinstance(value, str) for value in values[~null]):
            _fail(
                "estimation.cuped_linear.covariate_kind",
                name=name,
                kind="categorical values must be strings",
            )
        columns.append(
            tuple(
                None if absent else str(value) for value, absent in zip(values, null, strict=True)
            )
        )
    rows = tuple(zip(*columns, strict=True)) if columns else ((),) * len(groups)
    patterns = tuple(sorted(set(rows), key=_pattern_key))
    lookup = {pattern: index for index, pattern in enumerate(patterns)}
    codes = np.fromiter((lookup[row] for row in rows), dtype=int, count=len(groups))
    counts = np.zeros((len(order), len(patterns)), dtype=np.int64)
    references = np.zeros((len(outcomes), len(order)))
    exponents = [[0] * len(order) for _ in outcomes]
    first = np.zeros((len(order), len(outcomes), len(patterns)))
    scatter = np.zeros((len(order), len(outcomes), len(outcomes), len(patterns)))
    for a, arm_rows in enumerate(indices):
        if not len(arm_rows):
            continue
        grouped_rows = [[] for _ in patterns]
        for row in arm_rows:
            grouped_rows[codes[row]].append(row)
        bases = []
        shifts = []
        for f, values in enumerate(outcomes):
            arm_values = values[arm_rows]
            maximum = float(np.max(np.abs(arm_values)))
            if maximum:
                max_exponent = math.frexp(maximum)[1]
                scaled_maximum = math.ldexp(maximum, -max_exponent)
                scaled_squares = math.fsum((float(value) / maximum) ** 2 for value in arm_values)
                norm_scale = scaled_maximum * math.sqrt(scaled_squares)
                exponents[f][a] = min(1024, max_exponent + math.frexp(norm_scale)[1])
            offset, centered = _arm_offset(arm_values)
            references[f, a] = offset + _arm_mean(centered)
            bases.append(offset)
            shifts.append(references[f, a] - offset)
        for k, grouped in enumerate(grouped_rows):
            selected = np.asarray(grouped, dtype=int)
            count = len(selected)
            counts[a, k] = count
            if not count:
                continue
            centered = np.array(
                [
                    np.ldexp(values[selected], -exponents[f][a])
                    - math.ldexp(bases[f], -exponents[f][a])
                    for f, values in enumerate(outcomes)
                ]
            )
            sums = np.array([math.fsum(values) for values in centered])
            first[a, :, k] = [
                math.fsum((sums[f], -count * math.ldexp(shifts[f], -exponents[f][a])))
                for f in range(len(outcomes))
            ]
            centered -= sums[:, None] / count
            scatter[a, :, :, k] = centered @ centered.T
    levels = tuple(
        tuple(sorted(set(column), key=lambda value: (value is None, value or "")))
        for column in columns
    )
    return CategoricalCells(
        identity=identity,
        arms=order,
        patterns=patterns,
        counts=_freeze_payload(counts),
        levels=levels,
        null_metadata=tuple(
            (missing.get(name, "refuse"), None in level)
            for name, level in zip(covariates, levels, strict=True)
        ),
        outcome_references=_freeze_payload(references),
        centered_first_moments=_freeze_payload(first),
        outcome_exponents=_freeze_payload(exponents),
        symmetric_cross_moments=_freeze_payload(scatter),
        covariate_names=tuple(covariates),
    )


def select_cell_arms(cells: CategoricalCells, arms: Sequence[object]) -> CategoricalCells:
    """Evidence for a subset of arms; joint patterns and levels follow the kept units."""
    if not arms or any(arm not in cells.arms for arm in arms):
        _fail(
            "estimation.cuped_linear.invalid_arms",
            control=arms[0] if arms else None,
            arm_order=tuple(arms),
            observed=cells.arms,
        )
    index = [cells.arms.index(arm) for arm in arms]
    kept = [k for k in range(len(cells.patterns)) if any(cells.counts[a][k] for a in index)]
    patterns = tuple(cells.patterns[k] for k in kept)
    levels = tuple(
        tuple(
            sorted(
                {pattern[column] for pattern in patterns},
                key=lambda value: (value is None, value or ""),
            )
        )
        for column in range(len(cells.levels))
    )
    components = range(len(cells.outcome_references))
    return CategoricalCells(
        identity=cells.identity,
        arms=tuple(arms),
        patterns=patterns,
        counts=_freeze_payload([[cells.counts[a][k] for k in kept] for a in index]),
        levels=levels,
        null_metadata=tuple(
            (metadata[0], None in level)
            for metadata, level in zip(cells.null_metadata, levels, strict=True)
        ),
        outcome_references=_freeze_payload(
            [[cells.outcome_references[f][a] for a in index] for f in components]
        ),
        centered_first_moments=_freeze_payload(
            [
                [[cells.centered_first_moments[a][f][k] for k in kept] for f in components]
                for a in index
            ]
        ),
        symmetric_cross_moments=_freeze_payload(
            [
                [
                    [[cells.symmetric_cross_moments[a][f][g][k] for k in kept] for g in components]
                    for f in components
                ]
                for a in index
            ]
        ),
        covariate_names=cells.covariate_names,
        outcome_exponents=_freeze_payload(
            [[cells.outcome_exponents[f][a] for a in index] for f in components]
        ),
    )


def _scatter_root(scatter):
    values, vectors = np.linalg.eigh(scatter)
    tolerance = 64 * np.finfo(float).eps * np.max(np.abs(scatter), axis=(1, 2))
    if np.any(values[:, 0] < -tolerance):
        _invalid_evidence("categorical outcome scatter is not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0.0))[:, None, :]


def _denominator_support(cells, counts):
    support = []
    for a, count in enumerate(counts):
        first = np.asarray(cells.centered_first_moments[a][1])
        mean = (
            math.ldexp(cells.outcome_references[1][a], -cells.outcome_exponents[1][a])
            + math.fsum(first) / count
        )
        if mean <= 0:
            _fail(
                "estimation.cuped_linear.nonpositive_component",
                family="denominator",
                arm=cells.arms[a],
                value=math.ldexp(mean, cells.outcome_exponents[1][a]),
            )
        cell_counts = np.asarray(cells.counts[a])
        present = cell_counts > 0
        cell_means = first[present] / cell_counts[present]
        centered = cell_means - math.fsum(first) / count
        scatter = math.fsum(cells.symmetric_cross_moments[a][1][1]) + float(
            (centered * centered) @ cell_counts[present]
        )
        stats = ArmStats(
            study_id="cuped_linear",
            metric="ratio",
            group_id=str(cells.arms[a]),
            n=count,
            ref_y=0.0,
            cy1=0.0,
            cy2=0.0,
            ref_den=mean,
            cden1=0.0,
            cden2=scatter,
            cyden=0.0,
        )
        support.append(ratio_denominator_precision(stats))
    return tuple(support)


def fit_categorical_cells(cells: CategoricalCells, *, control) -> MulticovariateCupedFit:
    """Fit from complete joint cells, preserving original-unit HC2 and BM terms."""
    components = len(cells.outcome_references)
    if components not in (1, 2) or len(cells.covariate_names) != len(cells.levels):
        _invalid_evidence(
            "categorical fitting requires named joint covariates and one or two outcome components"
        )
    counts = tuple(sum(row) for row in cells.counts)
    observed = tuple(arm for arm, count in zip(cells.arms, counts, strict=True) if count)
    if len(cells.arms) < 2 or control not in cells.arms or observed != cells.arms:
        _fail(
            "estimation.cuped_linear.invalid_arms",
            control=control,
            arm_order=cells.arms,
            observed=observed,
        )
    if min(counts) < 2:
        _fail(
            "estimation.cuped_linear.arm_counts", counts=dict(zip(cells.arms, counts, strict=True))
        )
    for metadata, levels in zip(cells.null_metadata, cells.levels, strict=True):
        if (
            len(metadata) != 2
            or metadata[0] not in ("refuse", "impute-indicator")
            or metadata[1] != (None in levels)
        ):
            _invalid_evidence("categorical null metadata is invalid")
    positions = [
        (a, k) for a, row in enumerate(cells.counts) for k, count in enumerate(row) if count
    ]
    groups = np.array([cells.arms[a] for a, _ in positions], dtype=object)
    multiplicities = np.array([cells.counts[a][k] for a, k in positions])
    indices = tuple(np.flatnonzero(groups == arm) for arm in cells.arms)
    covariates = {
        name: np.array([cells.patterns[k][column] for _, k in positions], dtype=object)
        for column, name in enumerate(cells.covariate_names)
    }
    missing = {
        name: row[0] for name, row in zip(cells.covariate_names, cells.null_metadata, strict=True)
    }
    x, layout, offsets, scales = _encode_covariates(
        groups,
        indices,
        cells.arms,
        covariates,
        dict.fromkeys(covariates, "categorical"),
        missing,
        multiplicities,
    )
    position_exponents = np.array(
        [[cells.outcome_exponents[f][a] for a, _ in positions] for f in range(components)],
        dtype=int,
    )
    outcomes = tuple(
        np.ldexp(
            np.array(
                [cells.centered_first_moments[a][f][k] / cells.counts[a][k] for a, k in positions]
            ),
            position_exponents[f],
        )
        for f in range(components)
    )
    cell_cross = np.asarray(cells.symmetric_cross_moments, dtype=float)
    scatter = np.array([cell_cross[a, :, :, k] for a, k in positions])
    denominator_support = _denominator_support(cells, counts) if components == 2 else ()
    calculator, leverage = _fit_calculator(
        groups,
        cells.arms,
        counts,
        outcomes,
        x,
        layout,
        control,
        np.zeros((len(cells.arms), len(scales))),
        multiplicities=multiplicities,
        scatter_factor=(_scatter_root(scatter), position_exponents),
        denominator_support=denominator_support,
        outcome_references=cells.outcome_references,
    )
    identity = cells.identity + (
        "categorical-fit-v1",
        control,
        cells.arms,
        layout,
        cells.patterns,
        cells.counts,
        cells.outcome_references,
        cells.centered_first_moments,
        cells.symmetric_cross_moments,
        cells.outcome_exponents,
    )
    state = FitState(
        identity=identity,
        eligible_arms=cells.arms,
        withheld_arms=(),
        refusal_contexts=(),
        counts=counts,
        layout=layout,
        offsets=offsets,
        scales=scales,
        centered_means=calculator._x_means,
        anchor=calculator.anchor,
        triangular_factor=calculator._design_factor,
        transformed_rhs=tuple(
            calculator._design_q.T
            @ (np.sqrt(calculator._weights) * multiplicities * np.concatenate(component))
            for component in calculator._components
        ),
        slopes=calculator._slopes,
        adjusted_means=calculator.adjusted_means,
        theta=calculator.theta,
        n_min=min(counts),
        p=calculator.p,
        is_ratio=components == 2,
        denominator_support=tuple(zip(cells.arms, denominator_support, strict=True))
        if components == 2
        else (),
        control=control,
        within_arm_scatter=calculator._x_scatter,
    )
    cached = _cache_projections(calculator, outcomes, cells.arms)
    summary = _make_inference_summary(
        identity, cached, calculator, outcomes, cells.arms, leverage, _mean_covariance(calculator)
    )
    return finalize_fit(state, summary)
