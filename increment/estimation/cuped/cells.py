"""Parameter-independent joint categorical moments and exact compact fitting."""

from __future__ import annotations

import math

import numpy as np

from increment.estimation.armstats import ArmStats
from increment.estimation.cuped._numeric import _arm_mean, _arm_offset, _fail, _two_product
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
    _incompatible_evidence,
    _invalid_evidence,
)
from increment.estimation.variance import ratio_denominator_precision

# Partition centering, unpartitioned centering, stored roundings and the merge
# arithmetic together err by at most about ten units in the last place of each
# arm's absolute component sum.
_MERGE_ERROR = 2.0**-49
_MERGE_TOLERANCE = 2.0**-36
_MERGE_ROUTE = (
    "Fit from the unpartitioned cells, or explicitly supplied per-unit data, "
    "instead of merging these partitions."
)


def _pattern_key(pattern):
    return tuple((value is None, "" if value is None else value) for value in pattern)


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
    magnitudes = [[0.0] * len(order) for _ in outcomes]
    first = np.zeros((len(order), len(outcomes), len(patterns)))
    scatter = np.zeros((len(order), len(outcomes), len(outcomes), len(patterns)))
    for a, arm_rows in enumerate(indices):
        if not len(arm_rows):
            continue
        bases = []
        for f, values in enumerate(outcomes):
            arm_values = values[arm_rows]
            exponents[f][a] = math.frexp(float(np.max(np.abs(arm_values))))[1]
            magnitudes[f][a] = math.fsum(np.abs(np.ldexp(arm_values, -exponents[f][a])))
            offset, centered = _arm_offset(arm_values)
            references[f, a] = offset + _arm_mean(centered)
            bases.append(offset)
        # Center exactly as the per-unit fitter does, then rebase the exact cell
        # sums once onto the transferred arm-mean reference.
        shifts = [
            math.ldexp(references[f, a], -exponents[f][a]) - math.ldexp(bases[f], -exponents[f][a])
            for f in range(len(outcomes))
        ]
        for k in range(len(patterns)):
            selected = arm_rows[codes[arm_rows] == k]
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
                math.fsum((total, -count * shift))
                for total, shift in zip(sums, shifts, strict=True)
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
        outcome_magnitudes=_freeze_payload(magnitudes),
        symmetric_cross_moments=_freeze_payload(scatter),
        covariate_names=tuple(covariates),
    )


def _merge_cells(left: CategoricalCells, right: CategoricalCells) -> CategoricalCells:
    left_key = (
        left.identity,
        left.arms,
        left.covariate_names,
        tuple(row[0] for row in left.null_metadata),
        len(left.outcome_references),
    )
    right_key = (
        right.identity,
        right.arms,
        right.covariate_names,
        tuple(row[0] for row in right.null_metadata),
        len(right.outcome_references),
    )
    if left_key != right_key:
        _incompatible_evidence(
            "cannot merge categorical cells with incompatible source or method identities"
        )
    patterns = tuple(sorted(set(left.patterns) | set(right.patterns), key=_pattern_key))
    lookups = [{pattern: k for k, pattern in enumerate(part.patterns)} for part in (left, right)]
    cross_moments = [
        np.asarray(part.symmetric_cross_moments, dtype=float) for part in (left, right)
    ]
    components = len(left.outcome_references)
    arm_count = len(left.arms)
    exponents = [
        [
            max(
                (
                    part.outcome_exponents[f][a]
                    for part in (left, right)
                    if part.outcome_references[f][a]
                    or any(part.centered_first_moments[a][f])
                    or any(part.symmetric_cross_moments[a][f][f])
                ),
                default=0,
            )
            for a in range(arm_count)
        ]
        for f in range(components)
    ]
    counts = np.zeros((arm_count, len(patterns)), dtype=np.int64)
    references = np.zeros((components, arm_count))
    magnitudes = [[0.0] * arm_count for _ in range(components)]
    first = np.zeros((arm_count, components, len(patterns)))
    scatter = np.zeros((arm_count, components, components, len(patterns)))
    labels = ("numerator", "denominator") if components == 2 else ("outcome",)
    for a in range(arm_count):
        parts = [part for part in (left, right) if sum(part.counts[a])]
        if not parts:
            continue
        units = sum(sum(part.counts[a]) for part in parts)
        means = []
        for f in range(components):
            exponent = exponents[f][a]
            total = math.fsum(_arm_sum_terms(parts, a, f, exponent))
            magnitude = math.fsum(
                math.ldexp(part.outcome_magnitudes[f][a], part.outcome_exponents[f][a] - exponent)
                for part in parts
            )
            if len(parts) == 2 and _MERGE_ERROR * magnitude > _MERGE_TOLERANCE * abs(total):
                _fail(
                    "estimation.cuped_linear.merge_precision",
                    arm=left.arms[a],
                    component=labels[f],
                    bound=math.ldexp(_MERGE_ERROR * magnitude / units, exponent),
                    mean=math.ldexp(total / units, exponent),
                    route=_MERGE_ROUTE,
                )
            means.append(total / units)
            references[f, a] = math.ldexp(means[f], exponent)
            magnitudes[f][a] = magnitude
        for k, pattern in enumerate(patterns):
            entries = []
            for part, lookup, part_cross in zip((left, right), lookups, cross_moments, strict=True):
                old = lookup.get(pattern)
                if old is None or not part.counts[a][old]:
                    continue
                count = part.counts[a][old]
                shifts = [part.outcome_exponents[f][a] - exponents[f][a] for f in range(components)]
                offset = np.array(
                    [
                        math.fsum(
                            (
                                math.ldexp(part.outcome_references[f][a], -exponents[f][a]),
                                -means[f],
                                math.ldexp(
                                    part.centered_first_moments[a][f][old] / count, shifts[f]
                                ),
                            )
                        )
                        for f in range(components)
                    ]
                )
                scale = np.asarray(shifts)
                cross = np.ldexp(part_cross[a, :, :, old], scale[:, None] + scale[None, :])
                entries.append((part, old, count, offset, cross))
            if not entries:
                continue
            counts[a, k] = sum(entry[2] for entry in entries)
            for f in range(components):
                terms = []
                for part, old, count, _offset, _cross in entries:
                    reference = math.ldexp(part.outcome_references[f][a], -exponents[f][a])
                    terms.extend(_two_product(count, reference))
                    terms.append(
                        math.ldexp(
                            part.centered_first_moments[a][f][old],
                            part.outcome_exponents[f][a] - exponents[f][a],
                        )
                    )
                terms.extend(_two_product(-int(counts[a, k]), means[f]))
                first[a, f, k] = math.fsum(terms)
            cross = entries[0][4]
            if len(entries) == 2:
                (_, _, left_count, left_offset, left_cross) = entries[0]
                (_, _, right_count, right_offset, right_cross) = entries[1]
                share = math.sqrt(left_count * (right_count / (left_count + right_count)))
                weighted = share * (right_offset - left_offset)
                cross = left_cross + right_cross + np.outer(weighted, weighted)
            scatter[a, :, :, k] = cross
    levels = tuple(
        tuple(sorted(set(a) | set(b), key=lambda value: (value is None, value or "")))
        for a, b in zip(left.levels, right.levels, strict=True)
    )
    return CategoricalCells(
        identity=left.identity,
        arms=left.arms,
        patterns=patterns,
        counts=_freeze_payload(counts),
        levels=levels,
        null_metadata=tuple(
            (metadata[0], None in level)
            for metadata, level in zip(left.null_metadata, levels, strict=True)
        ),
        outcome_references=_freeze_payload(references),
        centered_first_moments=_freeze_payload(first),
        symmetric_cross_moments=_freeze_payload(scatter),
        covariate_names=left.covariate_names,
        outcome_exponents=_freeze_payload(exponents),
        outcome_magnitudes=_freeze_payload(magnitudes),
    )


def _arm_sum_terms(parts, arm, component, exponent):
    """Exact terms whose sum is an arm's component total at the merged scale."""
    terms = []
    for part in parts:
        reference = math.ldexp(part.outcome_references[component][arm], -exponent)
        shift = part.outcome_exponents[component][arm] - exponent
        for old, count in enumerate(part.counts[arm]):
            if count:
                terms.extend(_two_product(count, reference))
                terms.append(math.ldexp(part.centered_first_moments[arm][component][old], shift))
    return terms


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
        cells.outcome_magnitudes,
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
    )
    cached = _cache_projections(calculator, outcomes, cells.arms)
    summary = _make_inference_summary(
        identity, cached, calculator, outcomes, cells.arms, leverage, _mean_covariance(calculator)
    )
    return finalize_fit(state, summary)
