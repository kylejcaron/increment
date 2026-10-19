"""Stable factorization of a retained, scaled warehouse design relation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import ibis
import ibis.expr.types as ir
import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.linalg import solve_triangular

from increment.estimation.cuped._numeric import _fail
from increment.estimation.cuped.multicovariate import M1_MAX_LEVERAGE
from increment.estimation.cuped.types import FitState, InferenceSummary, ProjectionSpec
from increment.query._cuped_cells import _rows

FloatMatrix = tuple[tuple[float, ...], ...]


def _sum_sql(expressions: Iterable[Any]) -> Any:
    total = ibis.literal(0.0)
    for expression in expressions:
        total = total + expression
    return total


@dataclass(frozen=True, slots=True)
class QRFactor:
    """Bounded triangular factor and transformed outcome right-hand sides."""

    triangular_factor: tuple[tuple[float, ...], ...]
    transformed_rhs: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class QRUnitSummary:
    """Warehouse-resident row quantities and bounded leverage reductions."""

    relation: ir.Table
    max_leverage: float
    sum_leverage: float
    refusal_witness: tuple[tuple[str, object, float], ...]


def solve_qr_coefficients(
    factor: QRFactor,
    *,
    design_scales: Sequence[float],
    outcome_scales: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Solve the small triangular systems and restore input column scales."""
    triangular = np.asarray(factor.triangular_factor, dtype=float)
    transformed = np.asarray(factor.transformed_rhs, dtype=float)
    design_scale = np.asarray(design_scales, dtype=float)
    outcome_scale = np.asarray(outcome_scales, dtype=float)
    solved = solve_triangular(triangular, transformed, lower=False, check_finite=False)
    coefficients = solved * outcome_scale[None, :] / design_scale[:, None]
    return tuple(
        tuple(float(value) for value in coefficients[:, component])
        for component in range(coefficients.shape[1])
    )


def warehouse_qr_unit_summary(
    relation: ir.Table,
    factor: QRFactor,
    *,
    design_columns: Sequence[str],
    outcome_columns: Sequence[str],
    sqrt_weight_column: str,
    row_identity_columns: Sequence[str],
    design_scales: Sequence[float],
    coefficients: Sequence[Sequence[float]],
    execute: Callable[[ir.Table], object],
    scratch,
) -> QRUnitSummary:
    """Reconstruct leverage and regression residuals without exporting rows."""
    design_columns = tuple(design_columns)
    outcome_columns = tuple(outcome_columns)
    row_identity_columns = tuple(row_identity_columns)
    inverse_transpose = solve_triangular(
        np.asarray(factor.triangular_factor, dtype=float).T,
        np.eye(len(design_columns)),
        lower=True,
        check_finite=False,
    )
    q_columns = tuple(f"_qr_q_{index}" for index in range(len(design_columns)))
    coefficient_rows = [
        {
            "_qr_q_index": index,
            **{
                f"_qr_inverse_{design_index}": float(inverse_transpose[index, design_index])
                for design_index in range(len(design_columns))
            },
        }
        for index in range(len(design_columns))
    ]
    coefficient_table = scratch.materialize_next(ibis.memtable(coefficient_rows))
    products = relation.cross_join(coefficient_table)
    q_value = _sum_sql(
        products[design_columns[index]] * products[f"_qr_inverse_{index}"]
        for index in range(len(design_columns))
    )
    products = scratch.materialize_next(
        products.select(
            *row_identity_columns,
            "_qr_q_index",
            _qr_q_value=q_value,
        )
    )
    wide = products.group_by(*row_identity_columns).aggregate(
        **{
            name: products._qr_q_value.max(where=products._qr_q_index == index)
            for index, name in enumerate(q_columns)
        }
    )
    wide = scratch.materialize_next(wide)
    working = relation.join(wide, row_identity_columns)
    leverage = _sum_sql(working[name] * working[name] for name in q_columns)
    working = working.mutate(_qr_h=leverage)
    design_scales = tuple(float(value) for value in design_scales)
    residuals = {}
    for component, outcome_column in enumerate(outcome_columns):
        prediction = _sum_sql(
            working[name]
            * ibis.literal(design_scales[index] * float(coefficients[component][index]))
            / working[sqrt_weight_column]
            for index, name in enumerate(design_columns)
        )
        residuals[f"_qr_residual_{component}"] = working[outcome_column] - prediction
    working = working.mutate(**residuals)
    evidence = scratch.materialize_next(working)
    record = _rows(
        execute(
            evidence.aggregate(
                _qr_max_leverage=evidence._qr_h.max(),
                _qr_sum_leverage=evidence._qr_h.sum(),
            )
        )
    )[0]
    maximum = float(record["_qr_max_leverage"])
    total = float(record["_qr_sum_leverage"])
    witness = (
        (("max_leverage", maximum, float(M1_MAX_LEVERAGE)),) if maximum > M1_MAX_LEVERAGE else ()
    )
    return QRUnitSummary(evidence, maximum, total, witness)


def _aggregate_matrix(
    relation: ir.Table,
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    execute: Callable[[ir.Table], object],
    square_weight: Any | None = None,
    max_statement_chars: int = 80_000,
) -> FloatMatrix | tuple[FloatMatrix, FloatMatrix]:
    """Reduce a matrix in bounded one-row warehouse transfers."""
    rows = len(left)
    columns = len(right)
    first = np.zeros((rows, columns), dtype=float)
    second = np.zeros((rows, columns), dtype=float) if square_weight is not None else None
    fields_per_row = columns * (2 if second is not None else 1)
    row_batch = max(1, max_statement_chars // max(1, fields_per_row * 64))
    for start in range(0, rows, row_batch):
        stop = min(start + row_batch, rows)
        fields = {}
        for row_index in range(start, stop):
            for column_index, right_value in enumerate(right):
                product = left[row_index] * right_value
                fields[f"_qr_m_{row_index}_{column_index}"] = product.sum()
                if second is not None:
                    fields[f"_qr_w_{row_index}_{column_index}"] = (square_weight * product).sum()
        record = _rows(execute(relation.aggregate(**fields)))[0]
        for row_index in range(start, stop):
            for column_index in range(columns):
                first[row_index, column_index] = float(record[f"_qr_m_{row_index}_{column_index}"])
                if second is not None:
                    second[row_index, column_index] = float(
                        record[f"_qr_w_{row_index}_{column_index}"]
                    )
    first_result = tuple(tuple(float(value) for value in row) for row in first)
    if second is None:
        return first_result
    second_result = tuple(tuple(float(value) for value in row) for row in second)
    return first_result, second_result


def _stable_offset_difference(value: float, reference: float, scale: float) -> float:
    exponent = max(
        (math.frexp(abs(item))[1] for item in (value, reference, scale) if item != 0.0),
        default=0,
    )
    numerator = math.fsum((math.ldexp(value, -exponent), -math.ldexp(reference, -exponent)))
    return numerator / math.ldexp(scale, -exponent)


def _contrast_covariates(state: FitState, jacobian: np.ndarray) -> np.ndarray:
    arm_count = len(state.eligible_arms)
    p = state.p
    offsets = np.asarray(state.offsets, dtype=float)
    scales = np.asarray(state.scales, dtype=float)
    centered_means = np.asarray(state.centered_means, dtype=float)
    anchor = np.asarray(state.anchor, dtype=float)
    counts = tuple(state.counts)
    total = sum(counts)
    result = []
    for index in range(p):
        coordinate_offsets = tuple(
            _stable_offset_difference(offsets[index, arm], offsets[index, 0], scales[index])
            for arm in range(arm_count)
        )
        offset_anchor = math.fsum(
            counts[arm] / total * coordinate_offsets[arm] for arm in range(arm_count)
        )
        anchor_coordinate = (
            _stable_offset_difference(anchor[index], offsets[index, 0], scales[index])
            - offset_anchor
        )
        result.append(
            math.fsum(
                jacobian[arm]
                * (
                    centered_means[arm, index]
                    - anchor_coordinate
                    + coordinate_offsets[arm]
                    - offset_anchor
                )
                for arm in range(arm_count)
            )
        )
    return np.asarray(result, dtype=float)


def _warehouse_hc2_scores(
    state: FitState,
    relation: ir.Table,
    factor: QRFactor,
    design_scales: np.ndarray,
    sqrt_weight_column: str,
    covariate_columns: Sequence[str],
    execute: Callable[[ir.Table], object],
    scratch,
) -> tuple[ir.Table, tuple[tuple[str, ...], ...], tuple[tuple[float, ...], ...]]:
    arms = tuple(state.eligible_arms)
    counts = tuple(state.counts)
    width = len(arms) + state.p
    families = len(state.slopes)
    sqrt_weight = relation[sqrt_weight_column]
    q = tuple(relation[f"_qr_q_{index}"] for index in range(width))
    root_residual = (ibis.literal(1.0) - relation._qr_h).sqrt()
    slopes = np.asarray(state.slopes, dtype=float)
    covariate_fitted = tuple(
        _sum_sql(
            relation[covariate_columns[index]] * float(slopes[family, index])
            for index in range(state.p)
        )
        for family in range(families)
    )
    score_names = tuple(
        tuple(f"_qr_hc2_{family}_{arm}" for arm in range(len(arms))) for family in range(families)
    )
    score_values = {}
    triangular = np.asarray(factor.triangular_factor, dtype=float)
    for family in range(families):
        residual = relation[f"_qr_residual_{family}"]
        for arm in range(len(arms)):
            jacobian = np.zeros(len(arms), dtype=float)
            jacobian[arm] = 1.0
            contrast = np.concatenate((jacobian, -_contrast_covariates(state, jacobian)))
            coefficient = solve_triangular(
                triangular.T,
                contrast / design_scales,
                lower=True,
                check_finite=False,
            )
            influence = sqrt_weight * _sum_sql(
                q[index] * float(coefficient[index]) for index in range(width)
            )
            score_values[score_names[family][arm]] = (
                influence * residual / root_residual
                + float(jacobian.sum() / sum(counts)) * covariate_fitted[family]
            )
    scored = cast(ir.Table, scratch.materialize_next(relation.mutate(**score_values)))
    flat_scores = tuple(scored[name] for family in score_names for name in family)
    covariance = cast(
        FloatMatrix, _aggregate_matrix(scored, flat_scores, flat_scores, execute=execute)
    )
    return scored, score_names, covariance


def _warehouse_bm_component(
    current: ir.Table,
    base_columns: Sequence[str],
    contrast: np.ndarray,
    common_scale: float,
    factor: QRFactor,
    design_scales: np.ndarray,
    sqrt_weight_column: str,
    q_columns: Sequence[str],
    width: int,
    execute: Callable[[ir.Table], object],
    scratch,
) -> tuple[ir.Table, float, float, FloatMatrix, FloatMatrix]:
    if common_scale == 0.0:
        zero_matrix = tuple(tuple(0.0 for _ in range(2 * width)) for _ in range(2 * width))
        return current, 0.0, 0.0, zero_matrix, zero_matrix
    coefficient = solve_triangular(
        np.asarray(factor.triangular_factor, dtype=float).T,
        (contrast / common_scale) / design_scales,
        lower=True,
        check_finite=False,
    )
    q = tuple(current[name] for name in q_columns)
    sqrt_weight = current[sqrt_weight_column]
    root_residual = (ibis.literal(1.0) - current._qr_h).sqrt()
    t = (
        sqrt_weight
        * _sum_sql(q[index] * float(coefficient[index]) for index in range(width))
        / root_residual
    )
    feature_names = tuple(f"_qr_f_{index}" for index in range(2 * width))
    features = tuple(t * sqrt_weight * value for value in q) + tuple(
        t * value / sqrt_weight for value in q
    )
    current = scratch.materialize_next(
        current.select(*base_columns).mutate(
            _qr_t2=t.pow(2),
            **dict(zip(feature_names, features, strict=True)),
        )
    )
    record = _rows(
        execute(
            current.aggregate(
                _qr_s2=current._qr_t2.sum(),
                _qr_s4=current._qr_t2.pow(2).sum(),
            )
        )
    )[0]
    feature = tuple(current[name] for name in feature_names)
    gram, weighted_gram = cast(
        tuple[FloatMatrix, FloatMatrix],
        _aggregate_matrix(
            current,
            feature,
            feature,
            execute=execute,
            square_weight=current._qr_t2,
        ),
    )
    return (
        current,
        float(record["_qr_s2"]),
        float(record["_qr_s4"]),
        gram,
        weighted_gram,
    )


def _warehouse_projection_reductions(
    state: FitState,
    current: ir.Table,
    score_names: Sequence[Sequence[str]],
    projection_specs: Sequence[ProjectionSpec],
    projection_estimates: Sequence[float],
    factor: QRFactor,
    design_scales: np.ndarray,
    sqrt_weight_column: str,
    execute: Callable[[ir.Table], object],
    scratch,
) -> tuple[
    tuple[tuple[str, object, object, str, float, float, float], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
    tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
]:
    arms = tuple(state.eligible_arms)
    width = len(arms) + state.p
    q_columns = tuple(f"_qr_q_{index}" for index in range(width))
    base_columns = tuple(
        name
        for name in current.columns
        if not name.startswith("_qr_t") and not name.startswith("_qr_f")
    )
    records = []
    s2_rows = []
    s4_rows = []
    g_rows = []
    g2_rows = []
    for spec, estimate in zip(projection_specs, projection_estimates, strict=True):
        identity = tuple(spec.identity)
        scale_name = str(identity[len(state.identity) + 3])
        component_count = 2 if spec.kind == "ratio" else 1
        jacobian = np.asarray(spec.jacobian, dtype=float)
        if not spec.available:
            zero_matrix = tuple(tuple(0.0 for _ in range(2 * width)) for _ in range(2 * width))
            records.append(
                (
                    spec.kind,
                    spec.arm,
                    spec.control,
                    scale_name,
                    estimate,
                    float("nan"),
                    float("nan"),
                )
            )
            s2_rows.append(tuple(0.0 for _ in range(component_count)))
            s4_rows.append(tuple(0.0 for _ in range(component_count)))
            g_rows.append(tuple(zero_matrix for _ in range(component_count)))
            g2_rows.append(tuple(zero_matrix for _ in range(component_count)))
            continue
        actual_jacobian = (
            jacobian * (float(spec.scale) / float(spec.divisor)) * np.ldexp(1.0, spec.exponent)
        )
        score = _sum_sql(
            current[score_names[family][arm]] * float(actual_jacobian[family * len(arms) + arm])
            for family in range(component_count)
            for arm in range(len(arms))
        )
        score_record = _rows(execute(current.aggregate(_qr_score_square=score.pow(2).sum())))[0]
        records.append(
            (
                spec.kind,
                spec.arm,
                spec.control,
                scale_name,
                estimate,
                float(np.sqrt(max(0.0, score_record["_qr_score_square"]))),
                1.0,
            )
        )
        contrasts = tuple(
            np.concatenate(
                (
                    jacobian[family * len(arms) : (family + 1) * len(arms)],
                    -_contrast_covariates(
                        state, jacobian[family * len(arms) : (family + 1) * len(arms)]
                    ),
                )
            )
            for family in range(component_count)
        )
        common_scale = max(float(np.max(np.abs(value), initial=0.0)) for value in contrasts)
        reductions = []
        for contrast in contrasts:
            current, s2, s4, gram, weighted_gram = _warehouse_bm_component(
                current,
                base_columns,
                contrast,
                common_scale,
                factor,
                design_scales,
                sqrt_weight_column,
                q_columns,
                width,
                execute,
                scratch,
            )
            reductions.append((current, s2, s4, gram, weighted_gram))
        s2_rows.append(tuple(item[1] for item in reductions))
        s4_rows.append(tuple(item[2] for item in reductions))
        g_rows.append(tuple(item[3] for item in reductions))
        g2_rows.append(tuple(item[4] for item in reductions))
    return (
        tuple(records),
        tuple(s2_rows),
        tuple(s4_rows),
        tuple(g_rows),
        tuple(g2_rows),
    )


def warehouse_inference_summary(
    state: FitState,
    unit_summary: QRUnitSummary,
    factor: QRFactor,
    *,
    design_scales: Sequence[float],
    sqrt_weight_column: str,
    covariate_columns: Sequence[str],
    projection_specs: Sequence[ProjectionSpec],
    projection_estimates: Sequence[float],
    execute: Callable[[ir.Table], object],
    scratch,
) -> InferenceSummary:
    """Reduce exact HC2 and weighted-Q BM evidence inside the warehouse."""
    design_scale_array = np.asarray(design_scales, dtype=float)
    projection_specs = tuple(projection_specs)
    projection_estimates = tuple(float(value) for value in projection_estimates)
    if len(projection_specs) != len(projection_estimates):
        raise ValueError("projection specification and estimate counts differ")
    relation = unit_summary.relation
    width = len(state.eligible_arms) + state.p
    q_columns = tuple(f"_qr_q_{index}" for index in range(width))
    scored, score_names, covariance = _warehouse_hc2_scores(
        state,
        relation,
        factor,
        design_scale_array,
        sqrt_weight_column,
        tuple(covariate_columns),
        execute,
        scratch,
    )
    q = tuple(scored[name] for name in q_columns)
    sqrt_weight = scored[sqrt_weight_column]
    u_gram = cast(
        FloatMatrix,
        _aggregate_matrix(
            scored,
            tuple(sqrt_weight * value for value in q),
            tuple(sqrt_weight * value for value in q),
            execute=execute,
        ),
    )
    records, s2, s4, gram, weighted_gram = _warehouse_projection_reductions(
        state,
        scored,
        score_names,
        projection_specs,
        projection_estimates,
        factor,
        design_scale_array,
        sqrt_weight_column,
        execute,
        scratch,
    )
    return InferenceSummary(
        fit_identity=state.identity,
        projection_identity=tuple(state.identity) + ("all-pairs-v1",),
        max_leverage=unit_summary.max_leverage,
        sum_leverage=unit_summary.sum_leverage,
        refusal_witness=unit_summary.refusal_witness,
        adjusted_component_covariance=covariance,
        scaled_projected_score_squares=records,
        s2=s2,
        s4=s4,
        g=gram,
        g2=weighted_gram,
        u=u_gram,
        evidence_identity=state.identity,
        projection_specs=projection_specs,
    )


def factor_design_relation(
    relation: ir.Table,
    *,
    design_columns: Sequence[str],
    rhs_columns: Sequence[str],
    order_columns: Sequence[str],
    execute: Callable[[ir.Table], object],
    scratch,
    covariate_start: int = 0,
    covariate_names: Sequence[str] | None = None,
) -> QRFactor:
    """Householder-factor a weighted, centered, column-scaled design in SQL.

    Rows are ordered by the declared stable unit identity. Each reflector is
    computed over the current column from its diagonal row onward; scalar
    reflector products remain SQL subqueries and only the final R/RHS cross the
    client boundary.
    """
    design_columns = tuple(design_columns)
    rhs_columns = tuple(rhs_columns)
    covariate_names = tuple(covariate_names or design_columns[covariate_start:])
    row_index = ibis.row_number().over(order_by=[relation[name] for name in order_columns])
    work = relation.mutate(_qr_row=row_index).select("_qr_row", *design_columns, *rhs_columns)
    work = scratch.materialize_next(work)
    width = len(design_columns)
    all_columns = (*design_columns, *rhs_columns)

    for column_index in range(width):
        column_name = design_columns[column_index]
        column = work[column_name]
        remaining = work._qr_row >= column_index
        diagonal = work._qr_row == column_index
        max_stats = work.filter(remaining).aggregate(
            _qr_max=column.abs().max(),
            _qr_leading=column.max(where=diagonal),
        )
        scaled = work.cross_join(max_stats)
        safe_maximum = ibis.ifelse(max_stats._qr_max > 0.0, max_stats._qr_max, 1.0)
        square_stats = scaled.aggregate(
            _qr_scaled_square_sum=(
                (scaled[column_name] / safe_maximum)
                .pow(2)
                .sum(where=scaled._qr_row >= column_index)
            )
        )
        norm_stats = max_stats.cross_join(square_stats).mutate(
            _qr_norm=ibis.ifelse(
                max_stats._qr_max > 0.0,
                max_stats._qr_max * square_stats._qr_scaled_square_sum.sqrt(),
                0.0,
            )
        )
        # This is -copysign(norm, leading): both signed-zero cases select -norm.
        norm_stats = norm_stats.mutate(
            _qr_alpha=ibis.ifelse(
                norm_stats._qr_leading < 0.0,
                norm_stats._qr_norm,
                -norm_stats._qr_norm,
            )
        )
        reflected = work.cross_join(norm_stats).mutate(
            _qr_v=ibis.ifelse(
                work._qr_row < column_index,
                0.0,
                work[column_name]
                - ibis.ifelse(work._qr_row == column_index, norm_stats._qr_alpha, 0.0),
            )
        )
        reductions = reflected.aggregate(
            _qr_v2=(reflected._qr_v * reflected._qr_v).sum(),
            _qr_alpha=reflected._qr_alpha.max(),
            **{
                f"_qr_dot_{index}": (reflected._qr_v * reflected[name]).sum()
                for index, name in enumerate(all_columns)
                if index >= column_index
            },
        )
        reductions = reductions.mutate(
            _qr_beta=ibis.ifelse(reductions._qr_v2 > 0.0, 2.0 / reductions._qr_v2, 0.0)
        )
        source_columns = reflected.select("_qr_row", *all_columns, "_qr_v")
        updated = source_columns.cross_join(reductions)
        changes = {
            name: updated[name]
            - reductions._qr_beta * updated._qr_v * reductions[f"_qr_dot_{index}"]
            for index, name in enumerate(all_columns)
            if index >= column_index
        }
        work = scratch.materialize_next(updated.mutate(**changes).select("_qr_row", *all_columns))

    final_fields = {
        f"_qr_r_{row}_{column}": work[design_columns[column]].max(where=work._qr_row == row)
        for row in range(width)
        for column in range(row, width)
    }
    final_fields.update(
        {
            f"_qr_rhs_{row}_{component}": work[rhs_columns[component]].max(
                where=work._qr_row == row
            )
            for row in range(width)
            for component in range(len(rhs_columns))
        }
    )
    record = _rows(execute(work.aggregate(**final_fields)))[0]
    factor = tuple(
        tuple(
            float(record[f"_qr_r_{row}_{column}"]) if column >= row else 0.0
            for column in range(width)
        )
        for row in range(width)
    )
    rhs = tuple(
        tuple(float(record[f"_qr_rhs_{row}_{component}"]) for component in range(len(rhs_columns)))
        for row in range(width)
    )
    _apply_pivoted_r_rank(factor, covariate_start, covariate_names)
    return QRFactor(factor, rhs)


def _apply_pivoted_r_rank(
    factor: tuple[tuple[float, ...], ...],
    start: int,
    names: tuple[str, ...],
) -> None:
    """Apply the local fitter's column-scaled pivoted-R rank threshold."""
    if not names:
        return
    block = np.asarray(factor, dtype=float)[start:, start:]
    scales = np.linalg.norm(block, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        deficient = tuple(names[index] for index, scale in enumerate(scales) if scale == 0.0)
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=deficient or names,
            partners=(),
        )
    scaled = block / scales[None, :]
    _, pivoted, pivots = scipy_qr(scaled, mode="economic", pivoting=True, check_finite=False)
    diagonal_squares = np.diag(pivoted) ** 2
    rank = int(np.count_nonzero(diagonal_squares > len(names) * np.finfo(float).eps))
    if rank < len(names):
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=(names[int(pivots[rank])],),
            partners=tuple(names[int(index)] for index in pivots[:rank]),
        )
