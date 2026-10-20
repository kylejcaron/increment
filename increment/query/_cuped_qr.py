"""Stable factorization of a retained, scaled warehouse design relation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
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
from increment.query._nonfinite import nonfinite_parts

FloatMatrix = tuple[tuple[float, ...], ...]


def _sum_sql(expressions: Iterable[Any]) -> Any:
    total = ibis.literal(0.0)
    for expression in expressions:
        total = total + expression
    return total


def _power_two_l2_scales(  # noqa: PLR0915
    relation: ir.Table,
    expressions: Sequence[Any],
    *,
    execute: Callable[[ir.Table], object],
    zero_as_zero: bool = False,
    max_statement_chars: int = 80_000,
    square_sums: list[float] | None = None,
) -> tuple[float, ...]:
    """Return power-of-two L2 scales, keeping row maxima inside SQL."""
    chunk_size = max(1, max_statement_chars // 4096)
    if len(expressions) > chunk_size:
        scales: list[float] = []
        accumulated_squares: list[float] = []
        for start in range(0, len(expressions), chunk_size):
            chunk_squares: list[float] | None = [] if square_sums is not None else None
            scales.extend(
                _power_two_l2_scales(
                    relation,
                    expressions[start : start + chunk_size],
                    execute=execute,
                    zero_as_zero=zero_as_zero,
                    max_statement_chars=max_statement_chars,
                    square_sums=chunk_squares,
                )
            )
            if chunk_squares is not None:
                accumulated_squares.extend(chunk_squares)
        if square_sums is not None:
            square_sums.extend(accumulated_squares)
        return tuple(scales)

    maximum_names = tuple(f"_qr_scale_max_{index}" for index in range(len(expressions)))
    square_names = tuple(f"_qr_scale_squares_{index}" for index in range(len(expressions)))
    maxima = relation.aggregate(
        **{
            name: expression.cast("float64").abs().max()
            for name, expression in zip(maximum_names, expressions, strict=True)
        }
    )
    scaled = relation.cross_join(maxima)
    moment_fields: dict[str, Any] = {}
    for index, (maximum_name, square_name, expression) in enumerate(
        zip(maximum_names, square_names, expressions, strict=True)
    ):
        numeric = expression.cast("float64")
        moment_fields[f"_qr_norm_max_{index}"] = scaled[maximum_name].max()
        moment_fields[square_name] = (
            (numeric / ibis.ifelse(scaled[maximum_name] > 0, scaled[maximum_name], 1.0))
            .pow(2)
            .sum()
        )
    moments = scaled.aggregate(**moment_fields)

    from sqlglot import exp

    from increment.query._cuped_cells import _component_exponent_parts

    backend = relation.get_backend()
    dialect = backend.name

    def cte_name(name: str) -> str:
        return exp.to_identifier(name).sql(dialect=dialect, identify=True)

    moments_sql = ibis.to_sql(moments, dialect=dialect)
    moment_schema = {
        **{f"_qr_norm_max_{index}": "float64" for index in range(len(expressions))},
        **dict.fromkeys(square_names, "float64"),
    }
    moment_table = ibis.table(moment_schema, name="_qr_moments")
    normalized_fields = {}
    normalized_schema = {}
    for index, square_name in enumerate(square_names):
        maximum = moment_table[f"_qr_norm_max_{index}"]
        square_sum = moment_table[square_name]
        maximum_exponent, scaled_norm, valid = _component_exponent_parts(
            maximum, square_sum, dialect=dialect
        )
        normalized_fields[f"_qr_max_exponent_{index}"] = maximum_exponent
        normalized_fields[f"_qr_scaled_norm_{index}"] = scaled_norm
        normalized_fields[f"_qr_valid_{index}"] = valid
        normalized_fields[f"_qr_zero_{index}"] = maximum == 0
        normalized_schema.update(
            {
                f"_qr_max_exponent_{index}": "int64",
                f"_qr_scaled_norm_{index}": "float64",
                f"_qr_valid_{index}": "boolean",
                f"_qr_zero_{index}": "boolean",
            }
        )
    normalized_query = ibis.to_sql(moment_table.select(**normalized_fields), dialect=dialect)

    normalized_table = ibis.table(normalized_schema, name="_qr_normalized")
    scale_fields = {}
    scale_schema = {}
    for index in range(len(expressions)):
        zero = normalized_table[f"_qr_zero_{index}"]
        exponent = ibis.ifelse(
            zero,
            0,
            (
                normalized_table[f"_qr_max_exponent_{index}"]
                + normalized_table[f"_qr_scaled_norm_{index}"].log2().floor()
                + 1
            ).clip(upper=1023),
        ).cast("int64")
        bounded_exponent = ibis.ifelse(
            normalized_table[f"_qr_valid_{index}"] | zero, exponent, 0
        ).cast("int64")
        scale_fields[f"scale_{index}"] = bounded_exponent
        scale_schema[f"scale_{index}"] = "int64"
        if zero_as_zero:
            scale_fields[f"zero_{index}"] = zero
            scale_schema[f"zero_{index}"] = "boolean"
    scale_query = ibis.to_sql(normalized_table.select(**scale_fields), dialect=dialect)

    scale_table = ibis.table(scale_schema, name="_qr_scales")
    if square_sums is not None:
        value_names = tuple(f"_qr_scale_value_{index}" for index in range(len(expressions)))
        values = relation.mutate(
            **{
                name: expression.cast("float64")
                for name, expression in zip(value_names, expressions, strict=True)
            }
        )
        joined = values.cross_join(scale_table)
        result_fields: dict[str, Any] = {
            f"scale_{index}": joined[f"scale_{index}"].min() for index in range(len(expressions))
        }
        result_schema = {f"scale_{index}": "int64" for index in range(len(expressions))}
        if zero_as_zero:
            result_fields.update(
                {
                    f"zero_{index}": joined[f"zero_{index}"].ifelse(1, 0).cast("int64").min()
                    for index in range(len(expressions))
                }
            )
            result_schema.update({f"zero_{index}": "int64" for index in range(len(expressions))})
        result_fields.update(
            {
                f"squares_{index}": (joined[name] / (2.0 ** joined[f"scale_{index}"])).pow(2).sum()
                for index, name in enumerate(value_names)
            }
        )
        result_schema.update({f"squares_{index}": "float64" for index in range(len(expressions))})
        result = joined.aggregate(**result_fields)
    else:
        result_fields: dict[str, Any] = {
            f"scale_{index}": scale_table[f"scale_{index}"].min()
            for index in range(len(expressions))
        }
        result_schema = {f"scale_{index}": "int64" for index in range(len(expressions))}
        if zero_as_zero:
            result_fields.update(
                {
                    f"zero_{index}": scale_table[f"zero_{index}"].ifelse(1, 0).cast("int64").min()
                    for index in range(len(expressions))
                }
            )
            result_schema.update({f"zero_{index}": "int64" for index in range(len(expressions))})
        result = scale_table.aggregate(**result_fields)

    result_sql = ibis.to_sql(result, dialect=dialect)
    sql = (
        f"WITH {cte_name('_qr_moments')} AS ({moments_sql}), "
        f"{cte_name('_qr_normalized')} AS ({normalized_query}), "
        f"{cte_name('_qr_scales')} AS ({scale_query}) {result_sql}"
    )
    query = backend.sql(sql, schema=ibis.schema(result_schema))
    row = _rows(execute(query))[0]
    scales = []
    for index in range(len(expressions)):
        if zero_as_zero and bool(row[f"zero_{index}"]):
            scales.append(0.0)
        else:
            exponent = int(row[f"scale_{index}"])
            scale = math.ldexp(1.0, exponent)
            scales.append(scale if scale else math.ldexp(1.0, -1074))
    if square_sums is not None:
        square_sums.extend(float(row[f"squares_{index}"]) for index in range(len(expressions)))
    return tuple(scales)


def _stable_arm_means(
    relation: ir.Table, expression: Any, *, execute: Callable[[ir.Table], object]
) -> dict[str, float]:
    """Compute arm means after scaling by each arm's L2-norm exponent."""
    from increment.query._cuped_cells import (
        _binary_normalize,
        _compensated,
        _component_exponent_parts,
        _join_on,
    )

    counts = relation.group_by("group_id").aggregate(_qr_mean_count=relation.count())
    maxima = relation.group_by("group_id").aggregate(_qr_mean_max=expression.abs().max())
    scaled = _join_on(relation, maxima, ("group_id",))
    moments = scaled.group_by("group_id").aggregate(
        _qr_mean_max=scaled._qr_mean_max.max(),
        _qr_mean_squares=(
            expression / ibis.ifelse(scaled._qr_mean_max > 0, scaled._qr_mean_max, 1.0)
        )
        .pow(2)
        .sum(),
    )
    max_exponent, scaled_norm, valid = _component_exponent_parts(
        moments._qr_mean_max, moments._qr_mean_squares
    )
    exponents = moments.mutate(
        _qr_mean_exponent=ibis.ifelse(
            valid,
            max_exponent + scaled_norm.log2().floor() + 1,
            0,
        ).cast("int64")
    )
    joined = _join_on(relation, exponents, ("group_id",))
    joined = _join_on(joined, counts, ("group_id",))
    normalized = joined.mutate(
        _qr_mean_norm=_binary_normalize(expression, joined._qr_mean_exponent),
        _qr_mean_sigma=2.0 ** (joined._qr_mean_count.cast("float64").log2().ceil() + 2),
    )
    total, _ = _compensated(normalized._qr_mean_norm, normalized._qr_mean_sigma)
    means = normalized.group_by("group_id").aggregate(
        _qr_mean_scaled=total / normalized.count(),
        _qr_mean_exponent=normalized._qr_mean_exponent.min(),
    )
    rows = _rows(execute(means))
    return {
        str(row["group_id"]): math.ldexp(
            float(row["_qr_mean_scaled"]), int(row["_qr_mean_exponent"])
        )
        for row in rows
    }


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


def warehouse_qr_unit_summary(  # noqa: PLR0913
    relation: ir.Table,
    factor: QRFactor,
    *,
    design_columns: Sequence[str],
    outcome_columns: Sequence[str],
    sqrt_weight_column: str,
    row_identity_columns: Sequence[str],
    design_scales: Sequence[float],
    arms: Sequence[object],
    outcome_arm_exponents: Sequence[Sequence[int]],
    centered_outcome_means: Sequence[Sequence[float]],
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
    products = products.select(
        *row_identity_columns,
        "_qr_q_index",
        _qr_q_value=q_value,
    )
    product_table = ibis.table(products.schema(), name="_qr_q_products")
    wide = product_table.group_by(*row_identity_columns).aggregate(
        **{
            name: product_table._qr_q_value.max(where=product_table._qr_q_index == index)
            for index, name in enumerate(q_columns)
        }
    )
    from sqlglot import exp

    backend = relation.get_backend()
    dialect = backend.name
    cte_name = exp.to_identifier("_qr_q_products").sql(dialect=dialect, identify=True)
    wide_sql = (
        f"WITH {cte_name} AS ({ibis.to_sql(products, dialect=dialect)}) "
        f"{ibis.to_sql(wide, dialect=dialect)}"
    )
    wide = scratch.materialize_next(backend.sql(wide_sql, schema=wide.schema()))
    working = relation.join(wide, row_identity_columns)
    leverage = _sum_sql(working[name] * working[name] for name in q_columns)
    working = working.mutate(_qr_h=leverage)
    from increment.query._cuped_cells import _binary_normalize

    design_scales = tuple(float(value) for value in design_scales)
    residuals = {}
    for component, outcome_column in enumerate(outcome_columns):
        arm_residuals = []
        for arm_index, arm in enumerate(arms):
            exponent = int(outcome_arm_exponents[component][arm_index])
            prediction = _binary_normalize(
                ibis.literal(float(centered_outcome_means[component][arm_index])), exponent
            ) + _sum_sql(
                working[name]
                * design_scales[index]
                * _binary_normalize(ibis.literal(float(coefficients[component][index])), exponent)
                / working[sqrt_weight_column]
                for index, name in enumerate(design_columns)
                if index >= len(arms)
            )
            normalized_outcome = _binary_normalize(working[outcome_column], exponent)
            arm_residuals.append(
                ibis.ifelse(working["group_id"] == arm, normalized_outcome - prediction, 0.0)
            )
        residuals[f"_qr_residual_{component}"] = _sum_sql(arm_residuals)
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
    extra_fields: Mapping[str, Any] | None = None,
    extra_results: dict[str, float] | None = None,
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
        if start == 0 and extra_fields:
            fields.update(extra_fields)
        record = _rows(execute(relation.aggregate(**fields)))[0]
        if start == 0 and extra_results is not None and extra_fields:
            extra_results.update({name: float(record[name]) for name in extra_fields})
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


def _aggregate_grouped_matrix(
    relation: ir.Table,
    group_column: str,
    values: Sequence[Any],
    *,
    execute: Callable[[ir.Table], object],
    max_statement_chars: int = 80_000,
) -> dict[str, FloatMatrix]:
    """Reduce all group matrices in bounded statements."""
    width = len(values)
    row_batch = max(1, max_statement_chars // max(1, width * 64))
    matrices: dict[str, np.ndarray] = {}
    for start in range(0, width, row_batch):
        stop = min(start + row_batch, width)
        fields = {
            f"_qr_m_{row}_{column}": (values[row] * values[column]).sum()
            for row in range(start, stop)
            for column in range(width)
        }
        records = _rows(execute(relation.group_by(group_column).aggregate(**fields)))
        for record in records:
            group = str(record[group_column])
            matrix = matrices.setdefault(group, np.zeros((width, width), dtype=float))
            for row in range(start, stop):
                for column in range(width):
                    matrix[row, column] = float(record[f"_qr_m_{row}_{column}"])
    return {
        group: tuple(tuple(float(value) for value in row) for row in matrix)
        for group, matrix in matrices.items()
    }


def _sum_scaled_parts(parts: Sequence[tuple[float, int]]) -> float:
    nonzero = tuple((value, exponent) for value, exponent in parts if value != 0.0)
    if not nonzero:
        return 0.0
    common_exponent = max(exponent for _, exponent in nonzero)
    mantissa = math.fsum(
        math.ldexp(value, exponent - common_exponent) for value, exponent in nonzero
    )
    try:
        return math.ldexp(mantissa, common_exponent)
    except OverflowError:
        return math.copysign(float("inf"), mantissa)


def _stable_offset_difference(value: float, reference: float, scale: float) -> float:
    exponent = max(
        (math.frexp(abs(item))[1] for item in (value, reference, scale) if item != 0.0),
        default=0,
    )
    numerator = math.fsum((math.ldexp(value, -exponent), -math.ldexp(reference, -exponent)))
    return numerator / math.ldexp(scale, -exponent)


def _normalize_projection_jacobian(
    terms: Sequence[tuple[int, Sequence[float], Sequence[float], float]],
    size: int,
) -> tuple[tuple[float, ...], float, int]:
    """Represent Jacobian ratios without forming an overflowing reciprocal."""
    parts = []
    for index, numerator, denominator, sign in terms:
        mantissa = sign
        exponent = 0
        for value in numerator:
            fraction, shift = math.frexp(value)
            mantissa *= fraction
            exponent += shift
        for value in denominator:
            fraction, shift = math.frexp(value)
            mantissa /= fraction
            exponent -= shift
        fraction, shift = math.frexp(mantissa)
        parts.append((index, fraction, exponent + shift))
    if not parts:
        return tuple(0.0 for _ in range(size)), 1.0, 0
    scale_exponent = max(exponent for _, _, exponent in parts)
    scale_mantissa = max(
        abs(mantissa) for _, mantissa, exponent in parts if exponent == scale_exponent
    )
    normalized = [0.0] * size
    for index, mantissa, exponent in parts:
        normalized[index] = math.ldexp(mantissa / scale_mantissa, exponent - scale_exponent)
    return tuple(normalized), scale_mantissa, scale_exponent


def _stable_ratio(numerator: float, denominator: float) -> float:
    numerator_mantissa, numerator_exponent = math.frexp(numerator)
    denominator_mantissa, denominator_exponent = math.frexp(denominator)
    try:
        return math.ldexp(
            numerator_mantissa / denominator_mantissa,
            numerator_exponent - denominator_exponent,
        )
    except OverflowError:
        return math.copysign(float("inf"), numerator_mantissa * denominator_mantissa)


def _product_ratio_parts(
    numerators: Sequence[float], denominators: Sequence[float], weight: float
) -> tuple[float, int]:
    mantissa = weight
    exponent = 0
    for value in numerators:
        fraction, shift = math.frexp(value)
        mantissa *= fraction
        exponent += shift
    for value in denominators:
        fraction, shift = math.frexp(value)
        mantissa /= fraction
        exponent -= shift
    mantissa, shift = math.frexp(mantissa)
    return mantissa, exponent + shift


def _scaled_score_summary(
    projection_mantissa: float,
    projection_exponent: int,
    normalized_score_scale: float,
    scaled_square_sum: float,
) -> tuple[float, float]:
    """Rebase score evidence so neither factor has to hold the physical scale."""
    if projection_mantissa == 0.0 or normalized_score_scale == 0.0 or scaled_square_sum == 0.0:
        return 0.0, 0.0
    mantissa, exponent = math.frexp(projection_mantissa * normalized_score_scale)
    exponent += projection_exponent
    se_mantissa, se_shift = math.frexp(mantissa * math.sqrt(scaled_square_sum))
    se_exponent = exponent + se_shift
    scale_exponent = max(-1074, min(1023, se_exponent - 1))
    try:
        score_scale = math.ldexp(1.0, scale_exponent)
    except OverflowError:
        return float("inf"), float("inf")
    square_exponent = 2 * (exponent - scale_exponent)
    try:
        square_sum = math.ldexp(mantissa * mantissa * scaled_square_sum, square_exponent)
        final_se = math.ldexp(se_mantissa, se_exponent)
    except OverflowError:
        return float("inf"), float("inf")
    if not math.isfinite(final_se):
        return float("inf"), float("inf")
    return score_scale, square_sum


def _relative_score_coefficients(state, spec, arms):
    means = np.asarray(state.adjusted_means, dtype=float)
    treatment = arms.index(spec.arm)
    control = arms.index(spec.control)
    terms = (
        (0, treatment, (1.0,), (float(means[treatment]),), 1.0),
        (0, control, (1.0,), (float(means[control]),), -1.0),
    )
    return {
        (family, arm): _product_ratio_parts(numerators, denominators, sign)
        for family, arm, numerators, denominators, sign in terms
    }


def _ratio_score_coefficients(state, spec, arms):
    """Return per-component ratio derivative mantissas and binary exponents."""
    scale_name = str(spec.identity[-1])
    means = np.asarray(state.adjusted_means, dtype=float).reshape(2, len(arms))
    ai, ci = arms.index(spec.arm), arms.index(spec.control)
    nt, nd = float(means[0, ai]), float(means[1, ai])
    nc, cd = float(means[0, ci]), float(means[1, ci])
    if scale_name == "log":
        terms = (
            (0, ai, (1.0,), (nt,), 1.0),
            (0, ci, (1.0,), (nc,), -1.0),
            (1, ai, (1.0,), (nd,), -1.0),
            (1, ci, (1.0,), (cd,), 1.0),
        )
    else:
        terms = (
            (0, ai, (1.0,), (nd,), 1.0),
            (1, ai, (nt,), (nd, nd), -1.0),
            (0, ci, (1.0,), (cd,), -1.0),
            (1, ci, (nc,), (cd, cd), 1.0),
        )
    return {
        (family, arm): _product_ratio_parts(numerators, denominators, sign)
        for family, arm, numerators, denominators, sign in terms
    }


def _absolute_score_coefficients(spec, arms):
    jacobian = tuple(float(value) for value in spec.jacobian)
    coefficients = {}
    for family in range(1):
        for arm in range(len(arms)):
            mantissa, exponent = _product_ratio_parts(
                (jacobian[family * len(arms) + arm], float(spec.scale)),
                (float(spec.divisor),),
                1.0,
            )
            if mantissa:
                coefficients[(family, arm)] = (mantissa, exponent + int(spec.exponent))
    return coefficients


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
    outcome_arm_exponents: Sequence[Sequence[int]],
    execute: Callable[[ir.Table], object],
    scratch,
) -> tuple[ir.Table, tuple[tuple[str, ...], ...], tuple[tuple[float, ...], ...]]:
    from increment.query._cuped_cells import _binary_normalize

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
            ibis.ifelse(
                relation["group_id"] == arm,
                _sum_sql(
                    relation[covariate_columns[index]]
                    * _binary_normalize(
                        ibis.literal(float(slopes[family, index])),
                        int(outcome_arm_exponents[family][arm_index]),
                    )
                    for index in range(state.p)
                ),
                0.0,
            )
            for arm_index, arm in enumerate(arms)
        )
        for family in range(families)
    )
    score_names = tuple(
        tuple(f"_qr_hc2_{family}_{arm}" for arm in range(len(arms))) for family in range(families)
    )
    score_values = {}
    triangular = np.asarray(factor.triangular_factor, dtype=float)
    for family in range(families):
        normalized_residual = relation[f"_qr_residual_{family}"]
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
                influence * normalized_residual / root_residual
                + float(jacobian.sum() / sum(counts)) * covariate_fitted[family]
            )
    scored = cast(ir.Table, scratch.materialize_next(relation.mutate(**score_values)))
    flat_scores = tuple(scored[name] for family in score_names for name in family)
    covariance_by_arm = _aggregate_grouped_matrix(scored, "group_id", flat_scores, execute=execute)
    covariance_parts = [[[] for _ in flat_scores] for _ in flat_scores]
    for source_index, source_arm in enumerate(arms):
        normalized_covariance = covariance_by_arm[str(source_arm)]
        for left in range(len(flat_scores)):
            left_family = left // len(arms)
            for right in range(len(flat_scores)):
                right_family = right // len(arms)
                exponent = int(outcome_arm_exponents[left_family][source_index]) + int(
                    outcome_arm_exponents[right_family][source_index]
                )
                covariance_parts[left][right].append((normalized_covariance[left][right], exponent))
    covariance = tuple(
        tuple(_sum_scaled_parts(covariance_parts[left][right]) for right in range(len(flat_scores)))
        for left in range(len(flat_scores))
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
    moments: dict[str, float] = {}
    feature = tuple(current[name] for name in feature_names)
    gram, weighted_gram = cast(
        tuple[FloatMatrix, FloatMatrix],
        _aggregate_matrix(
            current,
            feature,
            feature,
            execute=execute,
            square_weight=current._qr_t2,
            extra_fields={
                "_qr_s2": current._qr_t2.sum(),
                "_qr_s4": current._qr_t2.pow(2).sum(),
            },
            extra_results=moments,
        ),
    )
    return (
        current,
        moments["_qr_s2"],
        moments["_qr_s4"],
        gram,
        weighted_gram,
    )


def _bm_component_score(
    relation: ir.Table,
    contrast: np.ndarray,
    common_scale: float,
    factor: QRFactor,
    design_scales: np.ndarray,
    sqrt_weight_column: str,
    q_columns: Sequence[str],
    width: int,
) -> Any:
    coefficient = solve_triangular(
        np.asarray(factor.triangular_factor, dtype=float).T,
        (contrast / common_scale) / design_scales,
        lower=True,
        check_finite=False,
    )
    q = tuple(relation[name] for name in q_columns)
    sqrt_weight = relation[sqrt_weight_column]
    root_residual = (ibis.literal(1.0) - relation._qr_h).sqrt()
    return (
        sqrt_weight
        * _sum_sql(q[index] * float(coefficient[index]) for index in range(width))
        / root_residual
    )


def _aggregate_bm_batch(
    relation: ir.Table,
    jobs: Sequence[tuple[str, Sequence[str]]],
    *,
    execute: Callable[[ir.Table], object],
    max_statement_chars: int = 80_000,
) -> tuple[tuple[float, float, FloatMatrix, FloatMatrix], ...]:
    matrix_width = len(jobs[0][1])
    descriptors: list[tuple[int, str, int, int, Any]] = []
    for job_index, (t2_name, feature_names) in enumerate(jobs):
        t2 = relation[t2_name]
        descriptors.extend(
            (
                (job_index, "s2", 0, 0, t2.sum()),
                (job_index, "s4", 0, 0, t2.pow(2).sum()),
            )
        )
        for row, left_name in enumerate(feature_names):
            for column in range(row, matrix_width):
                product = relation[left_name] * relation[feature_names[column]]
                descriptors.append((job_index, "g", row, column, product.sum()))
                descriptors.append((job_index, "g2", row, column, (t2 * product).sum()))
    values: dict[tuple[int, str, int, int], list[float]] = {}
    batch_size = max(1, max_statement_chars // 90)
    for start in range(0, len(descriptors), batch_size):
        batch = descriptors[start : start + batch_size]
        fields = {
            f"_qr_bm_{index}": expression
            for index, (_job, _kind, _row, _column, expression) in enumerate(batch)
        }
        records = _rows(execute(relation.group_by("group_id").aggregate(**fields)))
        for index, (job, kind, row, column, _expression) in enumerate(batch):
            values[(job, kind, row, column)] = [
                float(record[f"_qr_bm_{index}"] or 0.0) for record in records
            ]
    results = []
    for job_index in range(len(jobs)):
        matrix = np.zeros((matrix_width, matrix_width), dtype=float)
        weighted = np.zeros_like(matrix)
        for row in range(matrix_width):
            for column in range(row, matrix_width):
                gram_value = math.fsum(values[(job_index, "g", row, column)])
                weighted_value = math.fsum(values[(job_index, "g2", row, column)])
                matrix[row, column] = matrix[column, row] = gram_value
                weighted[row, column] = weighted[column, row] = weighted_value
        results.append(
            (
                math.fsum(values[(job_index, "s2", 0, 0)]),
                math.fsum(values[(job_index, "s4", 0, 0)]),
                tuple(tuple(float(value) for value in row) for row in matrix),
                tuple(tuple(float(value) for value in row) for row in weighted),
            )
        )
    return tuple(results)


def _sum_scaled_square_parts(parts: Sequence[tuple[float, int]]) -> tuple[float, int]:
    nonzero = tuple((value, exponent) for value, exponent in parts if value > 0.0)
    if not nonzero:
        return 0.0, 0
    common_exponent = max(exponent for _, exponent in nonzero)
    scaled_sum = math.fsum(
        math.ldexp(value, 2 * (exponent - common_exponent)) for value, exponent in nonzero
    )
    return scaled_sum, common_exponent


def _warehouse_projection_reductions(
    state: FitState,
    current: ir.Table,
    score_names: Sequence[Sequence[str]],
    projection_specs: Sequence[ProjectionSpec],
    projection_estimates: Sequence[float],
    factor: QRFactor,
    design_scales: np.ndarray,
    outcome_arm_exponents: Sequence[Sequence[int]],
    sqrt_weight_column: str,
    execute: Callable[[ir.Table], object],
    scratch,
) -> tuple[
    tuple[tuple[str, object, object, str, float, float, float], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
    tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
    tuple[bool, ...],
]:
    arms = tuple(str(arm) for arm in state.eligible_arms)
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
    unavailable_specs = []
    flat_score_names = tuple(name for family in score_names for name in family)
    base_component_scales = _power_two_l2_scales(
        current,
        tuple(
            ibis.ifelse(current["group_id"] == arm, current[name], 0.0)
            for arm in arms
            for name in flat_score_names
        ),
        execute=execute,
        zero_as_zero=True,
    )
    base_scale_lookup = {
        (arm, name): base_component_scales[arm_index * len(flat_score_names) + score_index]
        for arm_index, arm in enumerate(arms)
        for score_index, name in enumerate(flat_score_names)
    }
    projection_fields = {}
    projection_score_meta: dict[tuple[int, int], tuple[str, int]] = {}
    for spec_index, spec in enumerate(projection_specs):
        if not spec.available:
            continue
        if spec.kind == "ratio":
            coefficients = _ratio_score_coefficients(state, spec, arms)
        elif spec.kind == "relative":
            coefficients = _relative_score_coefficients(state, spec, arms)
        else:
            coefficients = _absolute_score_coefficients(spec, arms)
        component_count = 2 if spec.kind == "ratio" else 1
        for source_index, source_arm in enumerate(arms):
            term_parts = []
            for (family, target_arm), (mantissa, derivative_exponent) in coefficients.items():
                if family >= component_count or mantissa == 0.0:
                    continue
                name = score_names[family][target_arm]
                component_scale = base_scale_lookup[(source_arm, name)]
                if component_scale == 0.0:
                    continue
                component_exponent = math.frexp(component_scale)[1] - 1
                term_parts.append(
                    (
                        name,
                        mantissa,
                        derivative_exponent
                        + int(outcome_arm_exponents[family][source_index])
                        + component_exponent,
                        component_scale,
                    )
                )
            if not term_parts:
                continue
            common_exponent = max(item[2] for item in term_parts)
            score_name = f"_qr_projection_score_{spec_index}_{source_index}"
            score = _sum_sql(
                (current[name] / component_scale) * math.ldexp(mantissa, exponent - common_exponent)
                for name, mantissa, exponent, component_scale in term_parts
            )
            projection_fields[score_name] = ibis.ifelse(
                current["group_id"] == source_arm, score, 0.0
            )
            projection_score_meta[(spec_index, source_index)] = (score_name, common_exponent)
    projection_score_squares = []
    if projection_fields:
        current = scratch.materialize_next(current.mutate(**projection_fields))
        projection_score_names = tuple(projection_fields)
        projection_scales = _power_two_l2_scales(
            current,
            tuple(current[name] for name in projection_score_names),
            execute=execute,
            zero_as_zero=True,
            square_sums=projection_score_squares,
        )
        projection_scale_lookup = {
            (spec_index, source_index): (
                projection_scales[index],
                projection_score_squares[index],
            )
            for index, (spec_index, source_index) in enumerate(projection_score_meta)
        }
    else:
        projection_scale_lookup = {}
    return _warehouse_bm_projection_reductions(
        state=state,
        current=current,
        projection_specs=projection_specs,
        projection_estimates=projection_estimates,
        factor=factor,
        design_scales=design_scales,
        outcome_arm_exponents=outcome_arm_exponents,
        sqrt_weight_column=sqrt_weight_column,
        execute=execute,
        scratch=scratch,
        arms=arms,
        width=width,
        q_columns=q_columns,
        base_columns=base_columns,
        projection_score_meta=projection_score_meta,
        projection_scale_lookup=projection_scale_lookup,
        records=records,
        s2_rows=s2_rows,
        s4_rows=s4_rows,
        g_rows=g_rows,
        g2_rows=g2_rows,
        unavailable_specs=unavailable_specs,
    )


def _warehouse_bm_projection_reductions(  # noqa: PLR0913, PLR0915
    *,
    state: FitState,
    current: ir.Table,
    projection_specs: Sequence[ProjectionSpec],
    projection_estimates: Sequence[float],
    factor: QRFactor,
    design_scales: np.ndarray,
    outcome_arm_exponents: Sequence[Sequence[int]],
    sqrt_weight_column: str,
    execute: Callable[[ir.Table], object],
    scratch,
    arms: Sequence[str],
    width: int,
    q_columns: Sequence[str],
    base_columns: Sequence[str],
    projection_score_meta: Mapping[tuple[int, int], tuple[str, int]],
    projection_scale_lookup: Mapping[tuple[int, int], tuple[float, float]],
    records: list[Any],
    s2_rows: list[Any],
    s4_rows: list[Any],
    g_rows: list[Any],
    g2_rows: list[Any],
    unavailable_specs: list[bool],
):
    bm_score_fields = {}
    bm_jobs: list[tuple[str, str, tuple[str, ...]]] = []
    bm_jobs_by_spec: dict[int, list[int | None]] = {}
    for spec_index, (spec, estimate) in enumerate(
        zip(projection_specs, projection_estimates, strict=True)
    ):
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
            unavailable_specs.append(True)
            continue
        unavailable_specs.append(False)
        arm_square_parts = []
        for source_index in range(len(arms)):
            score_meta = projection_score_meta.get((spec_index, source_index))
            if score_meta is None:
                continue
            _score_name, common_exponent = score_meta
            normalized_score_scale, score_sum = projection_scale_lookup[(spec_index, source_index)]
            if normalized_score_scale != 0.0:
                arm_square_parts.append(
                    (
                        score_sum,
                        common_exponent + math.frexp(normalized_score_scale)[1] - 1,
                    )
                )
        combined_square_sum, combined_exponent = _sum_scaled_square_parts(arm_square_parts)
        score_scale, scaled_square_sum = _scaled_score_summary(
            1.0, combined_exponent, 1.0, combined_square_sum
        )
        if not math.isfinite(score_scale) or not math.isfinite(scaled_square_sum):
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
            zero_matrix = tuple(tuple(0.0 for _ in range(2 * width)) for _ in range(2 * width))
            g_rows.append(tuple(zero_matrix for _ in range(component_count)))
            g2_rows.append(tuple(zero_matrix for _ in range(component_count)))
            unavailable_specs[-1] = True
            continue
        records.append(
            (
                spec.kind,
                spec.arm,
                spec.control,
                scale_name,
                estimate,
                score_scale,
                scaled_square_sum,
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
        job_indices: list[int | None] = []
        for contrast in contrasts:
            if common_scale == 0.0:
                job_indices.append(None)
                continue
            job_index = len(bm_jobs)
            score = _bm_component_score(
                current,
                contrast,
                common_scale,
                factor,
                design_scales,
                sqrt_weight_column,
                q_columns,
                width,
            )
            t2_name = f"_qr_bm_t2_{job_index}"
            score_name = f"_qr_bm_score_{job_index}"
            feature_names = tuple(f"_qr_bm_f_{job_index}_{index}" for index in range(2 * width))
            bm_score_fields[t2_name] = score.pow(2)
            bm_score_fields[score_name] = score
            bm_jobs.append((t2_name, score_name, feature_names))
            job_indices.append(job_index)
        bm_jobs_by_spec[spec_index] = job_indices
        zero_matrix = tuple(tuple(0.0 for _ in range(2 * width)) for _ in range(2 * width))
        s2_rows.append(tuple(0.0 for _ in range(component_count)))
        s4_rows.append(tuple(0.0 for _ in range(component_count)))
        g_rows.append(tuple(zero_matrix for _ in range(component_count)))
        g2_rows.append(tuple(zero_matrix for _ in range(component_count)))
    bm_results = ()
    if bm_score_fields:
        score_relation = scratch.materialize_next(
            current.select(*base_columns).mutate(**bm_score_fields)
        )
        bm_feature_fields = {}
        aggregate_jobs = []
        for t2_name, score_name, feature_names in bm_jobs:
            score = score_relation[score_name]
            sqrt_weight = score_relation[sqrt_weight_column]
            q = tuple(score_relation[name] for name in q_columns)
            features = tuple(score * sqrt_weight * value for value in q) + tuple(
                score * value / sqrt_weight for value in q
            )
            bm_feature_fields.update(dict(zip(feature_names, features, strict=True)))
            aggregate_jobs.append((t2_name, feature_names))
        bm_relation = scratch.materialize_next(
            score_relation.mutate(**bm_feature_fields).select(
                *base_columns,
                *(name for name, _, _ in bm_jobs),
                *(name for _, _, feature_names in bm_jobs for name in feature_names),
            )
        )
        bm_results = _aggregate_bm_batch(bm_relation, aggregate_jobs, execute=execute)
    for spec_index, job_indices in bm_jobs_by_spec.items():
        for component_index, job_index in enumerate(job_indices):
            if job_index is None:
                continue
            s2, s4, gram, weighted_gram = bm_results[job_index]
            s2_rows[spec_index] = (
                *s2_rows[spec_index][:component_index],
                s2,
                *s2_rows[spec_index][component_index + 1 :],
            )
            s4_rows[spec_index] = (
                *s4_rows[spec_index][:component_index],
                s4,
                *s4_rows[spec_index][component_index + 1 :],
            )
            g_rows[spec_index] = (
                *g_rows[spec_index][:component_index],
                gram,
                *g_rows[spec_index][component_index + 1 :],
            )
            g2_rows[spec_index] = (
                *g2_rows[spec_index][:component_index],
                weighted_gram,
                *g2_rows[spec_index][component_index + 1 :],
            )
    return (
        tuple(records),
        tuple(s2_rows),
        tuple(s4_rows),
        tuple(g_rows),
        tuple(g2_rows),
        tuple(unavailable_specs),
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
    outcome_arm_exponents: Sequence[Sequence[int]],
    scratch,
) -> InferenceSummary:
    from dataclasses import replace

    design_scale_array = np.asarray(design_scales, dtype=float)
    projection_specs = tuple(projection_specs)
    projection_estimates = tuple(float(value) for value in projection_estimates)
    assert len(projection_specs) == len(projection_estimates), (
        "projection specification and estimate counts differ"
    )
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
        outcome_arm_exponents,
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
    records, s2, s4, gram, weighted_gram, unavailable_specs = _warehouse_projection_reductions(
        state,
        scored,
        score_names,
        projection_specs,
        projection_estimates,
        factor,
        design_scale_array,
        outcome_arm_exponents,
        sqrt_weight_column,
        execute,
        scratch,
    )
    projection_specs = tuple(
        replace(spec, available=spec.available and not unavailable)
        for spec, unavailable in zip(projection_specs, unavailable_specs, strict=True)
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
        f"_qr_r_{row}_{column}": work[design_columns[column]]
        .max(where=work._qr_row == row)
        .fill_null(0.0)
        for row in range(width)
        for column in range(row, width)
    }
    final_fields.update(
        {
            f"_qr_rhs_{row}_{component}": work[rhs_columns[component]]
            .max(where=work._qr_row == row)
            .fill_null(0.0)
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
    scales = np.max(np.abs(block), axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        deficient = tuple(names[index] for index, scale in enumerate(scales) if scale == 0.0)
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=deficient or names,
            partners=(),
        )
    scaled = block / scales[None, :]
    column_norms = np.linalg.norm(scaled, axis=0)
    scaled /= column_norms[None, :]
    _, pivoted, pivots = scipy_qr(scaled, mode="economic", pivoting=True, check_finite=False)
    diagonal_squares = np.diag(pivoted) ** 2
    rank = int(np.count_nonzero(diagonal_squares > len(names) * np.finfo(float).eps))
    if rank < len(names):
        _fail(
            "estimation.cuped_linear.rank_deficiency",
            columns=(names[int(pivots[rank])],),
            partners=tuple(names[int(index)] for index in pivots[:rank]),
        )


@dataclass(frozen=True, slots=True)
class EncodedQRDesign:
    """Retained, arm-centered covariates and their local-encoder layout."""

    relation: ir.Table
    layout: Any
    columns: tuple[str, ...]
    offsets: tuple[tuple[float, ...], ...]
    scales: tuple[float, ...]
    centered_means: tuple[tuple[float, ...], ...]
    anchor: tuple[float, ...]


class _QRDesignEncoder:
    def __init__(
        self,
        relation: ir.Table,
        arms: tuple[str, ...],
        row_identity_columns: tuple[str, ...],
        execute: Callable[[ir.Table], object],
    ):
        self.transformed = relation
        self.dialect = relation.get_backend().name
        self.arms = arms
        self.row_identity_columns = row_identity_columns
        self.execute = execute
        self.encoded_columns: list[str] = []
        self.query_columns: list[str] = []
        self.offsets: list[tuple[float, ...]] = []
        self.scales: list[float] = []
        self.levels: list[tuple[object, ...] | None] = []
        self.references: list[object] = []
        self.structural: list[str] = []
        self.indicator_masks: dict[str, list[str]] = {}
        self.counter = 0

    def encode(self, declarations: tuple[Any, ...]) -> EncodedQRDesign:
        from increment.estimation.cuped.encoding import MISSING_CATEGORY, MulticovariateCupedLayout

        for declaration in declarations:
            name = str(declaration.name)
            kind = str(declaration.kind)
            policy = str(getattr(declaration, "missing", "refuse"))
            if kind not in ("continuous", "categorical"):
                _fail("estimation.cuped_linear.covariate_kind", name=name, kind=kind)
            if policy not in ("refuse", "impute-indicator"):
                _fail("estimation.cuped_linear.missing_refused", name=name)
            value = self.transformed[name]
            null = value.isnull()
            if kind == "categorical" and not value.type().is_string():
                _fail(
                    "estimation.cuped_linear.covariate_kind",
                    name=name,
                    kind="categorical values must be strings",
                )
            if kind == "continuous":
                self._continuous(name, policy, value, null)
            else:
                self._categorical(name, policy, value, null, MISSING_CATEGORY)
        self._shared_indicators()
        centered_means, anchors = self._centered_anchors()
        return EncodedQRDesign(
            self.transformed,
            MulticovariateCupedLayout(
                tuple(str(item.name) for item in declarations),
                tuple(str(item.kind) for item in declarations),
                tuple(self.levels[: len(declarations)]),
                tuple(self.references[: len(declarations)]),
                tuple(self.encoded_columns),
                tuple(self.structural),
                tuple(self.shared_indicators),
            ),
            tuple(self.query_columns),
            tuple(self.offsets),
            tuple(self.scales),
            centered_means,
            anchors,
        )

    def _continuous(self, name: str, policy: str, value: Any, null: Any) -> None:
        numeric = (
            value.ifelse(1, 0).cast("float64")
            if value.type().is_boolean()
            else value.cast("float64")
        )
        if value.type().is_floating():
            is_nan, is_inf = nonfinite_parts(numeric, dialect=self.dialect)
            nonfinite = is_inf & ~numeric.isnull()
            nonfinite_count = _rows(
                self.execute(
                    self.transformed.aggregate(
                        _qr_nonfinite=nonfinite.ifelse(1, 0).cast("int64").sum()
                    )
                )
            )[0]["_qr_nonfinite"]
            if nonfinite_count:
                _fail("estimation.cuped_linear.nonfinite", name=name, row=0)
            null = null | is_nan
        if policy == "refuse":
            missing_count = _rows(
                self.execute(
                    self.transformed.aggregate(_qr_missing=null.ifelse(1, 0).cast("int64").sum())
                )
            )[0]["_qr_missing"]
            if missing_count:
                _fail("estimation.cuped_linear.missing_refused", name=name)
        if policy == "impute-indicator":
            counts = _rows(
                self.execute(
                    self.transformed.group_by("group_id").aggregate(
                        _qr_missing=null.ifelse(1, 0).cast("int64").sum(),
                        _qr_count=self.transformed.count(),
                    )
                )
            )
            if any(int(row["_qr_missing"]) > 0 for row in counts):
                self.indicator_masks.setdefault(name, []).append(name)
        safe = ibis.ifelse(null, 0.0, numeric)
        anchor_by_arm = _stable_arm_means(self.transformed, cast(Any, safe), execute=self.execute)
        arm_offsets = tuple(anchor_by_arm.get(arm, 0.0) for arm in self.arms)
        raw_centered = ibis.ifelse(null, 0.0, numeric) - _sum_sql(
            ibis.ifelse(self.transformed["group_id"] == arm, offset, 0.0)
            for arm, offset in zip(self.arms, arm_offsets, strict=True)
        )
        from increment.query._cuped_cells import _component_exponent_parts

        def scale_exponent(centered):
            maximum = centered.abs().max()
            maxima = self.transformed.aggregate(_qr_centered_max=maximum)
            scaled = self.transformed.cross_join(maxima).mutate(
                _qr_norm=centered
                / ibis.ifelse(maxima._qr_centered_max > 0, maxima._qr_centered_max, 1.0)
            )
            stats = scaled.aggregate(
                _qr_norm_squares=scaled._qr_norm.pow(2).sum(),
                _qr_centered_max=maxima._qr_centered_max.max(),
                _qr_nonzero=(scaled._qr_norm.abs().max() > 0),
            )
            max_exponent, scaled_norm, valid = _component_exponent_parts(
                stats._qr_centered_max, stats._qr_norm_squares
            )
            scale_valid = valid | (stats._qr_centered_max == 0)
            exponent = ibis.ifelse(
                valid,
                (max_exponent + scaled_norm.log2().floor() + 1).clip(upper=1023),
                0,
            ).cast("int64")
            row = _rows(
                self.execute(
                    stats.select(
                        _qr_scale_exponent=exponent,
                        _qr_scale_valid=scale_valid,
                        _qr_nonzero=stats._qr_nonzero,
                    )
                )
            )[0]
            return (
                int(row["_qr_scale_exponent"]),
                bool(row["_qr_scale_valid"]),
                bool(row["_qr_nonzero"]),
            )

        exponent, valid, nonzero = scale_exponent(raw_centered)
        if not valid:
            arm_offsets = tuple(0.0 for _ in self.arms)
            raw_centered = safe
            exponent, valid, nonzero = scale_exponent(raw_centered)
        if not valid:
            _fail("estimation.cuped_linear.nonfinite", name=name, row=0)
        if not nonzero:
            self.structural.append(name)
            self.levels.append(None)
            self.references.append(0.0)
            return
        scale = math.ldexp(1.0, exponent)
        if scale == 0.0:
            scale = math.ldexp(1.0, -1074)
        self.levels.append(None)
        self.references.append(0.0)
        self.offsets.append(arm_offsets)
        self.scales.append(scale)
        query_name = f"_qr_covariate_{self.counter}"
        from increment.query._cuped_cells import _binary_normalize

        self.transformed = self.transformed.mutate(
            **{query_name: _binary_normalize(cast(Any, raw_centered), exponent)}
        )
        self.counter += 1
        self.query_columns.append(query_name)
        self.encoded_columns.append(name)

    def _categorical(
        self, name: str, policy: str, value: Any, null: Any, missing_category: object
    ) -> None:
        from increment.query._cuped_cells import general_qr_required

        values = _rows(self.execute(self.transformed.select(name).distinct()))
        observed = {None if row[name] is None else str(row[name]) for row in values}
        category_levels = tuple(sorted(value for value in observed if value is not None))
        has_missing = None in observed
        full_levels: tuple[object, ...] = (
            *category_levels,
            *((missing_category,) if has_missing else ()),
        )
        if has_missing and policy == "refuse":
            _fail("estimation.cuped_linear.missing_refused", name=name)
        counts = _rows(
            self.execute(
                self.transformed.group_by("group_id", name).aggregate(
                    _qr_count=self.transformed.count()
                )
            )
        )
        frequencies = dict.fromkeys(full_levels, 0)
        by_arm_level: dict[tuple[str, object], int] = {}
        for row in counts:
            label = None if row[name] is None else str(row[name])
            key = missing_category if label is None else label
            frequencies[key] = frequencies.get(key, 0) + int(row["_qr_count"])
            by_arm_level[(str(row["group_id"]), key)] = int(row["_qr_count"])
        if len(full_levels) > 200:
            from types import SimpleNamespace

            general_qr_required(SimpleNamespace(name=name), "categorical_levels", covariate=name)
        for level in full_levels:
            observed_counts = {arm: by_arm_level.get((arm, level), 0) for arm in self.arms}
            if any(count < 2 for count in observed_counts.values()):
                shown = "<typed-missing>" if level is missing_category else level
                _fail(
                    "estimation.cuped_linear.level_support",
                    column=name,
                    level=shown,
                    counts=observed_counts,
                )
        reference = max(
            full_levels,
            key=lambda level: (frequencies.get(level, 0), -full_levels.index(level)),
            default=None,
        )
        self.levels.append(full_levels)
        self.references.append(reference)
        if len(full_levels) <= 1:
            self.structural.append(name)
            return
        for level in full_levels:
            if level == reference:
                continue
            shown = "<typed-missing>" if level is missing_category else repr(level)
            query_name = f"_qr_covariate_{self.counter}"
            encoded = (
                null.ifelse(1, 0)
                if level is missing_category
                else (value.cast("string") == str(level)).ifelse(1, 0)
            ).cast("float64")
            self.transformed = self.transformed.mutate(**{query_name: encoded})
            self.query_columns.append(query_name)
            self.encoded_columns.append(f"{name}[{shown}]")
            self.offsets.append(tuple(0.0 for _ in self.arms))
            self.scales.append(1.0)
            self.counter += 1

    def _shared_indicators(self) -> None:
        shared = []
        for name in self.indicator_masks:
            value = self.transformed[name]
            mask = value.isnull()
            if value.type().is_floating():
                is_nan, _is_inf = nonfinite_parts(value.cast("float64"), dialect=self.dialect)
                mask = mask | is_nan
            for owners in shared:
                representative = self.transformed[owners[0]]
                representative_mask = representative.isnull()
                if representative.type().is_floating():
                    is_nan, _is_inf = nonfinite_parts(
                        representative.cast("float64"), dialect=self.dialect
                    )
                    representative_mask = representative_mask | is_nan
                differences = _rows(
                    self.execute(
                        self.transformed.aggregate(
                            _qr_different=(mask != representative_mask)
                            .ifelse(1, 0)
                            .cast("int64")
                            .sum()
                        )
                    )
                )[0]["_qr_different"]
                if int(differences or 0) == 0:
                    owners.append(name)
                    break
            else:
                shared.append([name])
        retained = []
        for owners in shared:
            indicator_name = f"missing[{','.join(owners)}]"
            query_name = f"_qr_covariate_{self.counter}"
            value = self.transformed[owners[0]]
            indicator_value = value.isnull()
            if value.type().is_floating():
                is_nan, _is_inf = nonfinite_parts(value.cast("float64"), dialect=self.dialect)
                indicator_value = indicator_value | is_nan
            counts = _rows(
                self.execute(
                    self.transformed.aggregate(
                        _qr_missing=indicator_value.ifelse(1, 0).cast("int64").sum(),
                        _qr_count=self.transformed.count(),
                    )
                )
            )[0]
            if int(counts["_qr_missing"] or 0) in (
                0,
                int(counts["_qr_count"] or 0),
            ):
                continue
            self.transformed = self.transformed.mutate(
                **{query_name: indicator_value.ifelse(1, 0).cast("float64")}
            )
            self.query_columns.append(query_name)
            self.encoded_columns.append(indicator_name)
            retained.append(tuple(owners))
            self.offsets.append(tuple(0.0 for _ in self.arms))
            self.scales.append(1.0)
            self.counter += 1
        self.shared_indicators = retained

    def _centered_anchors(self) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        if not self.query_columns:
            return (), ()
        means_rows = _rows(
            self.execute(
                self.transformed.group_by("group_id").aggregate(
                    **{
                        f"_qr_mean_{index}": self.transformed[column].mean()
                        for index, column in enumerate(self.query_columns)
                    },
                    _qr_n=self.transformed.count(),
                )
            )
        )
        means_by_arm = {str(row["group_id"]): row for row in means_rows}
        centered_means = tuple(
            tuple(float(means_by_arm[arm][f"_qr_mean_{index}"]) for arm in self.arms)
            for index in range(len(self.query_columns))
        )
        count_lookup = {str(row["group_id"]): int(row["_qr_n"]) for row in means_rows}
        total = sum(count_lookup[arm] for arm in self.arms)
        anchors = tuple(
            math.fsum(
                count_lookup[arm] / total * centered_means[index][arm_index]
                + count_lookup[arm] / total * self.offsets[index][arm_index] / self.scales[index]
                for arm_index, arm in enumerate(self.arms)
            )
            * self.scales[index]
            for index in range(len(self.query_columns))
        )
        return centered_means, anchors


def encode_qr_covariates(
    relation: ir.Table,
    declarations: Sequence[Any],
    *,
    arms: Sequence[str],
    row_identity_columns: Sequence[str],
    execute: Callable[[ir.Table], object] | None = None,
) -> EncodedQRDesign:
    """Encode declared covariates relationally using the local CUPED layout rules."""
    execute = execute or (lambda expression: expression.execute())
    encoder = _QRDesignEncoder(
        relation,
        tuple(str(arm) for arm in arms),
        tuple(row_identity_columns),
        execute,
    )
    return encoder.encode(tuple(declarations))


@dataclass(frozen=True, slots=True)
class QRPreparedRelation:
    """Weighted factor input plus bounded scale and outcome metadata."""

    relation: ir.Table
    design_columns: tuple[str, ...]
    rhs_columns: tuple[str, ...]
    outcome_columns: tuple[str, ...]
    covariate_columns: tuple[str, ...]
    sqrt_weight_column: str
    design_scales: tuple[float, ...]
    outcome_scales: tuple[float, ...]
    counts: tuple[int, ...]
    outcome_offsets: tuple[tuple[float, ...], ...]
    centered_outcome_means: tuple[tuple[float, ...], ...]
    outcome_arm_exponents: tuple[tuple[int, ...], ...]


def prepare_qr_factor_input(  # noqa: PLR0915
    encoded: EncodedQRDesign,
    *,
    outcome_columns: Sequence[str],
    arms: Sequence[str],
    row_identity_columns: Sequence[str],
    execute: Callable[[ir.Table], object],
    scratch,
    control: str | None = None,
) -> QRPreparedRelation:
    """Center, weight and max-scale the retained QR design and outcome RHS."""

    from increment.estimation.cuped._numeric import _fail
    from increment.query._cuped_cells import _rows

    relation = encoded.relation
    arms = tuple(str(arm) for arm in arms)
    outcome_columns = tuple(outcome_columns)
    row_identity_columns = tuple(row_identity_columns)
    counts_rows = _rows(execute(relation.group_by("group_id").aggregate(_qr_n=relation.count())))
    count_lookup = {str(row["group_id"]): int(row["_qr_n"]) for row in counts_rows}
    if set(count_lookup) != set(arms):
        _fail(
            "estimation.cuped_linear.invalid_arms",
            control=control if control is not None else (arms[0] if arms else None),
            arm_order=arms,
            observed=tuple(count_lookup),
        )
    counts = tuple(count_lookup.get(arm, 0) for arm in arms)
    if min(counts, default=0) < 2:
        from increment.estimation.cuped._numeric import _fail

        _fail(
            "estimation.cuped_linear.arm_counts",
            counts=dict(zip(arms, counts, strict=True)),
        )
    sqrt_weight = _sum_sql(
        ibis.ifelse(
            relation["group_id"] == arm,
            math.sqrt(1.0 / (count_lookup[arm] * (count_lookup[arm] - 1))),
            0.0,
        )
        for arm in arms
    )
    arm_columns = tuple(f"_qr_arm_design_{index}" for index in range(len(arms)))
    fields = {
        "group_id": relation["group_id"],
        **{name: relation[name] for name in row_identity_columns},
    }
    fields.update(
        {
            name: ibis.ifelse(relation["group_id"] == arm, sqrt_weight, 0.0)
            for name, arm in zip(arm_columns, arms, strict=True)
        }
    )
    covariate_columns = tuple(
        f"_qr_centered_covariate_{index}" for index in range(len(encoded.columns))
    )
    weighted_covariate_columns = tuple(
        f"_qr_weighted_covariate_{index}" for index in range(len(encoded.columns))
    )
    for index, (source_name, target_name, weighted_name) in enumerate(
        zip(encoded.columns, covariate_columns, weighted_covariate_columns, strict=True)
    ):
        arm_mean = _sum_sql(
            ibis.ifelse(
                relation["group_id"] == arm,
                float(encoded.centered_means[index][arm_index]),
                0.0,
            )
            for arm_index, arm in enumerate(arms)
        )
        centered = relation[source_name] - arm_mean
        fields[target_name] = centered
        fields[weighted_name] = centered * sqrt_weight
    offsets: list[tuple[float, ...]] = []
    centered_means: list[tuple[float, ...]] = []
    outcome_fields = {}
    for component, source_name in enumerate(outcome_columns):
        raw_value = relation[source_name]
        value = (
            raw_value.ifelse(1, 0).cast("float64")
            if raw_value.type().is_boolean()
            else raw_value.cast("float64")
        )
        is_nan, is_inf = nonfinite_parts(value, dialect=relation.get_backend().name)
        outcome_nonfinite = value.isnull() | is_nan | is_inf
        raw_stats = _rows(
            execute(
                relation.aggregate(
                    _qr_outcome_nonfinite=outcome_nonfinite.ifelse(1, 0).cast("int64").sum()
                )
            )
        )[0]
        if int(raw_stats["_qr_outcome_nonfinite"] or 0):
            _fail("estimation.cuped_linear.nonfinite", name=source_name, row=0)
        anchor_by_arm = _stable_arm_means(relation, value, execute=execute)
        component_offsets = tuple(anchor_by_arm[arm] for arm in arms)
        centered = value - _sum_sql(
            ibis.ifelse(relation["group_id"] == arm, offset, 0.0)
            for arm, offset in zip(arms, component_offsets, strict=True)
        )
        centered_nan, centered_inf = nonfinite_parts(centered, dialect=relation.get_backend().name)
        centered_nonfinite = _rows(
            execute(
                relation.aggregate(
                    _qr_centered_nonfinite=cast(Any, centered_nan | centered_inf)
                    .ifelse(1, 0)
                    .cast("int64")
                    .sum()
                )
            )
        )[0]["_qr_centered_nonfinite"]
        if int(centered_nonfinite or 0):
            component_offsets = tuple(0.0 for _ in arms)
            centered = value
            component_means = tuple(anchor_by_arm[arm] for arm in arms)
        else:
            component_means = tuple(0.0 for _ in arms)
        residual_name = f"_qr_outcome_centered_{component}"
        outcome_fields[residual_name] = centered
        offsets.append(component_offsets)
        centered_means.append(component_means)
    relation = relation.mutate(**fields, _qr_sqrt_weight=sqrt_weight, **outcome_fields)
    outcome_names = tuple(f"_qr_outcome_centered_{index}" for index in range(len(outcome_columns)))
    raw_design_columns = (*arm_columns, *weighted_covariate_columns)
    design_expressions = tuple(relation[name] for name in raw_design_columns)
    outcome_expressions = tuple(relation[outcome] for outcome in outcome_names)
    numeric_outcomes = tuple(
        relation[source_name].ifelse(ibis.literal(1), ibis.literal(0)).cast("float64")
        if relation[source_name].type().is_boolean()
        else relation[source_name]
        for source_name in outcome_columns
    )
    arm_outcome_expressions = tuple(
        ibis.ifelse(relation["group_id"] == arm, outcome, 0.0)
        for outcome in numeric_outcomes
        for arm in arms
    )
    all_scales = _power_two_l2_scales(
        relation,
        (*design_expressions, *outcome_expressions, *arm_outcome_expressions),
        execute=execute,
    )
    design_scale_count = len(design_expressions)
    outcome_scale_count = len(outcome_expressions)
    design_scales = all_scales[:design_scale_count]
    outcome_scales = all_scales[design_scale_count : design_scale_count + outcome_scale_count]
    arm_outcome_scales = all_scales[design_scale_count + outcome_scale_count :]
    outcome_arm_exponents = tuple(
        tuple(
            math.frexp(arm_outcome_scales[component * len(arms) + arm_index])[1] - 1
            for arm_index in range(len(arms))
        )
        for component in range(len(outcome_columns))
    )
    from increment.query._cuped_cells import _binary_normalize

    design_columns = tuple(f"_qr_design_{index}" for index in range(len(raw_design_columns)))
    rhs_names = tuple(f"_qr_rhs_{index}" for index in range(len(outcome_names)))
    relation = relation.mutate(
        **{
            name: _binary_normalize(relation[source_name], math.frexp(scale)[1] - 1)
            for name, source_name, scale in zip(
                design_columns, raw_design_columns, design_scales, strict=True
            )
        },
        **{
            rhs: (
                _binary_normalize(relation[outcome], math.frexp(scale)[1] - 1)
                * relation["_qr_sqrt_weight"]
            )
            for rhs, outcome, scale in zip(rhs_names, outcome_names, outcome_scales, strict=True)
        },
    )
    return QRPreparedRelation(
        relation,
        design_columns,
        rhs_names,
        outcome_names,
        covariate_columns,
        "_qr_sqrt_weight",
        design_scales,
        outcome_scales,
        counts,
        tuple(offsets),
        tuple(centered_means),
        outcome_arm_exponents,
    )


def fit_warehouse_qr(  # noqa: PLR0915
    relation: ir.Table,
    *,
    metric,
    request,
    execute: Callable[[ir.Table], object],
    scratch,
    return_components: bool = False,
    magnitude_checked: bool = False,
) -> Any:
    """Fit and finalize a mixed CUPED model from a retained warehouse relation."""
    from increment.estimation.cuped._numeric import _fail
    from increment.estimation.cuped.inference import finalize_fit
    from increment.estimation.cuped.types import FitState, ProjectionSpec
    from increment.query._cuped_cells import _rows

    arms = tuple(request.arm_order)
    if any(arm not in arms for arm in (request.control,)):
        _fail("estimation.cuped_linear.control_missing", control=request.control)
    relation = relation.filter(relation["group_id"].isin(arms))
    declarations = tuple(request.method.covariates or ())
    from increment.query._cuped_cells import require_warehouse_magnitude_range

    outcome_components = (
        (("numerator", "y"), ("denominator", "y_den"))
        if metric.type == "ratio"
        else (("outcome", "y"),)
    )
    if not magnitude_checked:
        require_warehouse_magnitude_range(
            relation,
            metric=metric.name,
            components=(
                *outcome_components,
                *(
                    (declaration.name, declaration.name)
                    for declaration in declarations
                    if declaration.kind == "continuous"
                ),
            ),
            execute=execute,
        )
    row_identity_columns = ("unit_id",)
    encoded = encode_qr_covariates(
        relation,
        declarations,
        arms=arms,
        row_identity_columns=row_identity_columns,
        execute=execute,
    )
    encoded = EncodedQRDesign(
        scratch.materialize_input(encoded.relation),
        encoded.layout,
        encoded.columns,
        encoded.offsets,
        encoded.scales,
        encoded.centered_means,
        encoded.anchor,
    )
    outcome_columns = ("y", "y_den") if metric.type == "ratio" else ("y",)
    prepared = prepare_qr_factor_input(
        encoded,
        outcome_columns=outcome_columns,
        arms=arms,
        row_identity_columns=row_identity_columns,
        execute=execute,
        scratch=scratch,
        control=request.control,
    )
    factor = factor_design_relation(
        prepared.relation,
        design_columns=prepared.design_columns,
        rhs_columns=prepared.rhs_columns,
        order_columns=row_identity_columns,
        execute=execute,
        scratch=scratch,
        covariate_start=len(arms),
        covariate_names=encoded.layout.encoded_columns,
    )
    coefficients = solve_qr_coefficients(
        factor,
        design_scales=prepared.design_scales,
        outcome_scales=prepared.outcome_scales,
    )
    arm_count = len(arms)
    slopes = tuple(tuple(row[arm_count:]) for row in coefficients)
    component_means = tuple(
        tuple(
            math.fsum(
                (
                    prepared.outcome_offsets[component][arm_index],
                    prepared.centered_outcome_means[component][arm_index],
                )
            )
            for arm_index in range(arm_count)
        )
        for component in range(len(outcome_columns))
    )
    adjusted_means = []
    for component in range(len(outcome_columns)):
        for arm_index in range(arm_count):
            difference = tuple(
                math.fsum(
                    (
                        encoded.centered_means[index][arm_index],
                        _stable_offset_difference(
                            encoded.offsets[index][arm_index],
                            encoded.anchor[index],
                            encoded.scales[index],
                        ),
                    )
                )
                for index in range(len(encoded.columns))
            )
            adjustment = math.fsum(
                slopes[component][index] * difference[index]
                for index in range(len(encoded.columns))
            )
            adjusted_means.append(component_means[component][arm_index] - adjustment)
    denominator_support = ()
    if metric.type == "ratio":
        from increment.query._cuped_cells import (
            _binary_normalize,
            _component_exponent_parts,
            _join_on,
        )

        denominator = relation["y_den"].cast("float64")
        maxima = relation.group_by("group_id").aggregate(_qr_den_max=denominator.abs().max())
        scaled = _join_on(relation, maxima, ("group_id",))
        moments = scaled.group_by("group_id").aggregate(
            _qr_den_max=scaled._qr_den_max.max(),
            _qr_den_squares=(
                denominator / ibis.ifelse(scaled._qr_den_max > 0, scaled._qr_den_max, 1.0)
            )
            .pow(2)
            .sum(),
        )
        max_exponent, scaled_norm, valid = _component_exponent_parts(
            moments._qr_den_max, moments._qr_den_squares
        )
        exponents = moments.mutate(
            _qr_den_exponent=ibis.ifelse(
                valid,
                max_exponent + scaled_norm.log2().floor() + 1,
                0,
            ).cast("int64")
        )
        normalized_rows = _join_on(relation, exponents, ("group_id",))
        normalized = cast(
            ir.Table,
            scratch.materialize_next(
                normalized_rows.select(
                    group_id=normalized_rows["group_id"],
                    _qr_den_norm=_binary_normalize(
                        denominator, normalized_rows["_qr_den_exponent"]
                    ),
                )
            ),
        )
        normalized_mean = normalized["_qr_den_norm"].mean().over(group_by=[normalized["group_id"]])
        centered_denominator = cast(
            ir.Table,
            scratch.materialize_next(
                normalized.mutate(_qr_den_residual=normalized["_qr_den_norm"] - normalized_mean)
            ),
        )
        moments = _rows(
            execute(
                centered_denominator.group_by("group_id").aggregate(
                    _qr_den_squares=centered_denominator["_qr_den_residual"].pow(2).sum(),
                    _qr_den_mean=centered_denominator["_qr_den_norm"].mean(),
                    _qr_den_positive=centered_denominator["_qr_den_norm"].mean() > 0,
                    _qr_den_count=centered_denominator.count(),
                )
            )
        )
        actual_denominator_means = _stable_arm_means(relation, denominator, execute=execute)
        support_rows = []
        for row in moments:
            arm = str(row["group_id"])
            if not bool(row["_qr_den_positive"]):
                _fail(
                    "estimation.cuped_linear.nonpositive_component",
                    family="denominator",
                    arm=arm,
                    value=actual_denominator_means[arm],
                )
            n = int(row["_qr_den_count"])
            precision = (
                math.sqrt(float(row["_qr_den_squares"]))
                / math.sqrt(n)
                / math.sqrt(n - 1)
                / float(row["_qr_den_mean"])
            )
            support_rows.append((arm, precision))
        denominator_support = tuple(support_rows)
    state = FitState(
        identity=request.identity,
        eligible_arms=arms,
        withheld_arms=(),
        refusal_contexts=(),
        counts=prepared.counts,
        layout=encoded.layout,
        offsets=encoded.offsets,
        scales=encoded.scales,
        centered_means=tuple(
            tuple(encoded.centered_means[index][arm_index] for index in range(len(encoded.columns)))
            for arm_index in range(arm_count)
        ),
        anchor=encoded.anchor,
        triangular_factor=factor.triangular_factor,
        transformed_rhs=tuple(
            tuple(
                factor.transformed_rhs[row][component] for row in range(len(factor.transformed_rhs))
            )
            for component in range(len(outcome_columns))
        ),
        slopes=slopes,
        adjusted_means=tuple(adjusted_means),
        theta=tuple(
            slope[index] / encoded.scales[index]
            for slope in slopes
            for index in range(len(encoded.scales))
        ),
        n_min=min(prepared.counts),
        p=len(encoded.columns),
        is_ratio=metric.type == "ratio",
        denominator_support=denominator_support,
        control=request.control,
    )
    specs = []
    estimates = []
    mean_matrix = np.asarray(state.adjusted_means, dtype=float).reshape(
        len(outcome_columns), arm_count
    )
    for treatment_index in range(1, arm_count):
        for control_index in range(treatment_index):
            treatment, control = arms[treatment_index], arms[control_index]
            for arm, comparison in ((treatment, control), (control, treatment)):
                jacobian = np.zeros(arm_count)
                jacobian[arms.index(arm)] = 1.0
                jacobian[arms.index(comparison)] = -1.0
                specs.append(
                    ProjectionSpec(
                        identity=state.identity + ("absolute", arm, comparison, ""),
                        kind="absolute",
                        jacobian=tuple(float(value) for value in jacobian),
                        scale=1.0,
                        divisor=1.0,
                        exponent=0,
                        arm=arm,
                        control=comparison,
                        available=True,
                    )
                )
                estimates.append(
                    float(mean_matrix[0, arms.index(arm)] - mean_matrix[0, arms.index(comparison)])
                )
            if np.all(mean_matrix[0, [treatment_index, control_index]] > 0):
                from increment.estimation.cuped.multicovariate import _LinearCupedCalculator

                for arm, comparison in ((treatment, control), (control, treatment)):
                    ai, ci = arms.index(arm), arms.index(comparison)
                    means = tuple(
                        (mantissa, 0.0, exponent)
                        for value in (mean_matrix[0, ai], mean_matrix[0, ci])
                        for mantissa, exponent in (math.frexp(float(value)),)
                    )
                    jacobian, scale, divisor, exponent = (
                        _LinearCupedCalculator._scaled_jacobian_details(
                            arm_count,
                            ((ai, means[0], 1.0), (ci, means[1], -1.0)),
                        )
                    )
                    specs.append(
                        ProjectionSpec(
                            identity=state.identity + ("relative", arm, comparison, ""),
                            kind="relative",
                            jacobian=tuple(float(value) for value in jacobian),
                            scale=scale,
                            divisor=divisor,
                            exponent=exponent,
                            arm=arm,
                            control=comparison,
                            available=True,
                        )
                    )
                    estimates.append(
                        float(math.log(mean_matrix[0, ai]) - math.log(mean_matrix[0, ci]))
                    )
            if len(outcome_columns) == 2 and np.all(
                mean_matrix[:, [treatment_index, control_index]] > 0
            ):
                for scale in ("log", "absolute"):
                    for arm, comparison in ((treatment, control), (control, treatment)):
                        ai, ci = arms.index(arm), arms.index(comparison)
                        nt, nd = float(mean_matrix[0, ai]), float(mean_matrix[1, ai])
                        nc, cd = float(mean_matrix[0, ci]), float(mean_matrix[1, ci])
                        if scale == "log":
                            jacobian_terms = (
                                (ai, (1.0,), (nt,), 1.0),
                                (ci, (1.0,), (nc,), -1.0),
                                (arm_count + ai, (1.0,), (nd,), -1.0),
                                (arm_count + ci, (1.0,), (cd,), 1.0),
                            )
                            estimate = math.log(nt) - math.log(nd) - math.log(nc) + math.log(cd)
                        else:
                            treatment_ratio = _stable_ratio(nt, nd)
                            control_ratio = _stable_ratio(nc, cd)
                            jacobian_terms = (
                                (ai, (1.0,), (nd,), 1.0),
                                (ci, (1.0,), (cd,), -1.0),
                                (arm_count + ai, (nt,), (nd, nd), -1.0),
                                (arm_count + ci, (nc,), (cd, cd), 1.0),
                            )
                            with np.errstate(over="ignore", invalid="ignore"):
                                estimate = treatment_ratio - control_ratio
                        jacobian_values, jacobian_scale, jacobian_exponent = (
                            _normalize_projection_jacobian(jacobian_terms, 2 * arm_count)
                        )
                        available = math.isfinite(float(estimate))
                        if not available:
                            jacobian_values = tuple(0.0 for _ in jacobian_values)
                            jacobian_scale, jacobian_exponent = 1.0, 0
                        specs.append(
                            ProjectionSpec(
                                identity=state.identity + ("ratio", arm, comparison, scale),
                                kind="ratio",
                                jacobian=jacobian_values,
                                scale=jacobian_scale,
                                divisor=1.0,
                                exponent=jacobian_exponent,
                                arm=arm,
                                control=comparison,
                                available=available,
                            )
                        )
                        estimates.append(float(estimate))
    unit_summary = warehouse_qr_unit_summary(
        prepared.relation,
        factor,
        design_columns=prepared.design_columns,
        outcome_columns=prepared.outcome_columns,
        sqrt_weight_column=prepared.sqrt_weight_column,
        row_identity_columns=row_identity_columns,
        design_scales=prepared.design_scales,
        arms=arms,
        outcome_arm_exponents=prepared.outcome_arm_exponents,
        centered_outcome_means=prepared.centered_outcome_means,
        coefficients=coefficients,
        execute=execute,
        scratch=scratch,
    )
    summary = warehouse_inference_summary(
        state,
        unit_summary,
        factor,
        design_scales=prepared.design_scales,
        sqrt_weight_column=prepared.sqrt_weight_column,
        covariate_columns=prepared.covariate_columns,
        projection_specs=tuple(specs),
        projection_estimates=tuple(estimates),
        outcome_arm_exponents=prepared.outcome_arm_exponents,
        execute=execute,
        scratch=scratch,
    )
    fit = finalize_fit(state, summary)
    return (state, summary, fit) if return_components else fit
