"""Bounded joint categorical evidence with warehouse-only reconstruction guards."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NoReturn, cast

import ibis
import ibis.expr.types as ir
import narwhals as nw
import numpy as np

from increment.errors import CapabilityError, RefusalSpec, refuse
from increment.estimation.cuped._numeric import _fail
from increment.estimation.cuped.cells import _pattern_key
from increment.estimation.cuped.types import CategoricalCells, _freeze_payload

if TYPE_CHECKING:
    from increment.query.cuped import _ResolvedCupedRequest

MAX_CELLS = 4096
MAX_ENCODED_BYTES = 8 * 1024 * 1024
MIN_COMPRESSION = 8
MIN_CONTRIBUTORS = 3
_SUM_ERROR = 2.0**-52
_MEAN_TOLERANCE = 2.0**-36

_GENERAL_QR = RefusalSpec(
    "query.cuped.general_qr_required",
    CapabilityError,
    template="Metric {metric!r} requires general warehouse QR ({reason}); {route}",
    keys=frozenset({"metric", "reason", "route"}),
)
REFUSALS = {_GENERAL_QR.code: _GENERAL_QR}

_MAGNITUDE_RANGE = RefusalSpec(
    "query.cuped.magnitude_range",
    CapabilityError,
    template=(
        "Metric {metric!r} has {component} in arm {arm!r} outside the warehouse CUPED "
        "RMS magnitude range [1e-50, 1e50] (bound: {bound}); rescale the values "
        "(change units) or fit from explicitly supplied per-unit data with "
        "Analysis.from_unit_summary or Analysis.from_unit_panel."
    ),
    keys=frozenset({"metric", "arm", "component", "bound"}),
)
_MAGNITUDE_MIN = 1e-50
_MAGNITUDE_MAX = 1e50
REFUSALS[_MAGNITUDE_RANGE.code] = _MAGNITUDE_RANGE


def general_qr_required(metric, reason: str, **detail: object) -> NoReturn:
    refuse(
        _GENERAL_QR,
        metric=metric.name,
        reason=reason,
        route="Use explicitly supplied local per-unit data with Analysis.from_unit_summary or Analysis.from_unit_panel; warehouse units are never downloaded automatically.",
        **detail,
    )


def require_warehouse_magnitude_range(  # noqa: PLR0915
    relation: ir.Table,
    *,
    metric: str,
    components: Sequence[tuple[str, str]],
    execute,
) -> None:
    names = tuple(name for _, name in components)
    if not names:
        return
    outcome_components = {"outcome", "numerator", "denominator"}
    component_values: list[Any] = []
    maxima_fields: dict[str, Any] = {}
    for index, (component, name) in enumerate(components):
        raw = relation[name]
        value = cast(
            ir.FloatingValue,
            raw.ifelse(ibis.literal(1), ibis.literal(0)).cast("float64")
            if raw.type().is_boolean()
            else raw.cast("float64"),
        )
        nonfinite = value.isinf()
        if component in outcome_components:
            nonfinite = nonfinite | value.isnan()
        else:
            value = cast(
                ir.FloatingValue,
                ibis.ifelse(value.isnan(), ibis.null().cast("float64"), value),
            )
        component_values.append(value)
        maxima_fields[f"_magnitude_nonfinite_{index}"] = cast(
            Any, nonfinite.ifelse(ibis.literal(1), ibis.literal(0)).cast("int64")
        ).sum()
        maxima_fields[f"_magnitude_max_{index}"] = cast(Any, value).abs().max()

    maxima = relation.group_by("group_id").aggregate(**maxima_fields)
    maxima_schema = {
        "group_id": relation.schema()["group_id"],
        **{f"_magnitude_nonfinite_{index}": "int64" for index in range(len(components))},
        **{f"_magnitude_max_{index}": "float64" for index in range(len(components))},
    }
    maxima_table = ibis.table(maxima_schema, name="_magnitude_maxima")
    joined = relation.join(
        maxima_table,
        [relation.group_id.identical_to(maxima_table.group_id)],
    )
    stats_fields: dict[str, Any] = {}
    for index, value in enumerate(component_values):
        maximum = maxima_table[f"_magnitude_max_{index}"]
        safe_maximum = (maximum > 0).ifelse(maximum, 1.0)
        stats_fields[f"_magnitude_nonfinite_{index}"] = maxima_table[
            f"_magnitude_nonfinite_{index}"
        ].max()
        stats_fields[f"_magnitude_max_{index}"] = maximum.max()
        stats_fields[f"_magnitude_squares_{index}"] = (value / safe_maximum).pow(2).sum()
        stats_fields[f"_magnitude_count_{index}"] = value.count()
    stats = joined.group_by("group_id").aggregate(**stats_fields)

    stats_schema = {
        "group_id": relation.schema()["group_id"],
        **{
            field: "int64"
            for index in range(len(components))
            for field in (
                f"_magnitude_nonfinite_{index}",
                f"_magnitude_count_{index}",
            )
        },
        **{
            field: "float64"
            for index in range(len(components))
            for field in (
                f"_magnitude_max_{index}",
                f"_magnitude_squares_{index}",
            )
        },
    }
    stats_table = ibis.table(stats_schema, name="_magnitude_stats")
    normalized_fields: dict[str, Any] = {}
    for index in range(len(components)):
        maximum_exponent, scaled_norm, valid = _component_exponent_parts(
            stats_table[f"_magnitude_max_{index}"],
            stats_table[f"_magnitude_squares_{index}"],
        )
        count = stats_table[f"_magnitude_count_{index}"].cast("float64")
        safe_count = (count > 0).ifelse(count, 1.0)
        rms_factor = scaled_norm / safe_count.sqrt()
        normalized_fields[f"_magnitude_nonfinite_{index}"] = stats_table[
            f"_magnitude_nonfinite_{index}"
        ]
        normalized_fields[f"_magnitude_max_exponent_{index}"] = maximum_exponent
        normalized_fields[f"_magnitude_rms_factor_{index}"] = rms_factor
        normalized_fields[f"_magnitude_valid_{index}"] = valid & (count > 0)
    normalized = stats_table.select("group_id", **normalized_fields)

    minimum_mantissa, minimum_exponent = math.frexp(_MAGNITUDE_MIN)
    maximum_mantissa, maximum_bound_exponent = math.frexp(_MAGNITUDE_MAX)
    normalized_schema = {
        "group_id": relation.schema()["group_id"],
        **{f"_magnitude_nonfinite_{index}": "int64" for index in range(len(components))},
        **{
            field: dtype
            for index in range(len(components))
            for field, dtype in (
                (f"_magnitude_max_exponent_{index}", "int64"),
                (f"_magnitude_rms_factor_{index}", "float64"),
                (f"_magnitude_valid_{index}", "boolean"),
            )
        },
    }
    normalized_table = ibis.table(normalized_schema, name="_magnitude_normalized")
    decision_fields: dict[str, Any] = {}
    for index in range(len(components)):
        factor = normalized_table[f"_magnitude_rms_factor_{index}"]
        adjustment = factor.log2().floor().cast("int64") + 1
        exponent = normalized_table[f"_magnitude_max_exponent_{index}"] + adjustment
        mantissa = _binary_normalize(factor, adjustment)
        valid = normalized_table[f"_magnitude_valid_{index}"] & (factor > 0)
        decision_fields[f"_magnitude_nonfinite_{index}"] = normalized_table[
            f"_magnitude_nonfinite_{index}"
        ]
        decision_fields[f"_magnitude_low_{index}"] = valid & (
            (exponent < minimum_exponent)
            | ((exponent == minimum_exponent) & (mantissa < minimum_mantissa))
        )
        decision_fields[f"_magnitude_high_{index}"] = valid & (
            (exponent > maximum_bound_exponent)
            | ((exponent == maximum_bound_exponent) & (mantissa > maximum_mantissa))
        )
    decisions = normalized_table.select("group_id", **decision_fields)

    from sqlglot import exp

    backend = relation.get_backend()
    dialect = backend.name

    def cte(name: str, expression: ir.Table) -> str:
        identifier = exp.to_identifier(name).sql(dialect=dialect, identify=True)
        return f"{identifier} AS ({ibis.to_sql(expression, dialect=dialect)})"

    sql = (
        "WITH "
        + ", ".join(
            (
                cte("_magnitude_maxima", maxima),
                cte("_magnitude_stats", stats),
                cte("_magnitude_normalized", normalized),
            )
        )
        + " "
        + ibis.to_sql(decisions, dialect=dialect)
    )
    output_schema = {
        "group_id": relation.schema()["group_id"],
        **{
            field: dtype
            for index in range(len(components))
            for field, dtype in (
                (f"_magnitude_nonfinite_{index}", "int64"),
                (f"_magnitude_low_{index}", "boolean"),
                (f"_magnitude_high_{index}", "boolean"),
            )
        },
    }
    result = backend.sql(sql, schema=ibis.schema(output_schema))
    summary_rows = _rows(execute(result))
    for row in summary_rows:
        for index, (_component, name) in enumerate(components):
            if row[f"_magnitude_nonfinite_{index}"]:
                _fail("estimation.cuped_linear.nonfinite", name=name, row=0)
    for row in summary_rows:
        for index, (component, _name) in enumerate(components):
            if row[f"_magnitude_low_{index}"]:
                refuse(
                    _MAGNITUDE_RANGE,
                    metric=metric,
                    arm=row["group_id"],
                    component=component,
                    bound="lower",
                )
            if row[f"_magnitude_high_{index}"]:
                refuse(
                    _MAGNITUDE_RANGE,
                    metric=metric,
                    arm=row["group_id"],
                    component=component,
                    bound="upper",
                )


def known_outcome_domain(metric):
    if metric.type not in ("mean", "ratio") or getattr(metric, "winsorization", None) is not None:
        return True
    measures = (metric.numerator, metric.denominator) if metric.type == "ratio" else (metric,)
    return any(
        getattr(measure, "aggregation", None) in ("count", "count_distinct", "any")
        for measure in measures
    )


def known_qr_outcome_domain(metric):
    """Whether QR is ruled out by declarations before inspecting source columns."""
    if metric.type not in ("mean", "ratio", "conversion"):
        return True
    return bool(getattr(getattr(metric, "winsorization", None), "has_percentile", False))


def _known_outcome_domain(metric, relation, outcomes):
    if known_outcome_domain(metric):
        return True
    return any(component_domain(None, relation[name].type()) != "continuous" for name in outcomes)


def component_domain(measure, datatype):
    """Retain the original outcome domain before numerical preparation."""
    if getattr(measure, "aggregation", None) in ("count", "count_distinct", "any"):
        return "discrete"
    if datatype is None or datatype.is_integer() or datatype.is_boolean() or datatype.is_decimal():
        return "discrete"
    return "continuous" if datatype.is_floating() else "unknown"


def _binary_normalize(value, exponent):
    half = exponent // 2
    return value.cast("float64") / (2.0**half) / (2.0 ** (exponent - half))


def _binary_denormalize(value, exponent):
    half = exponent // 2
    return value * (2.0**half) * (2.0 ** (exponent - half))


def _split_terms(value, sigma):
    """Split terms exactly into two grid-rounded parts and a final remainder.

    For ``count`` terms of magnitude at most two and ``sigma >= 4 * count``, every
    partial sum of each rounded part is exact in any order: the first part lies on
    the float grid at ``sigma``, the second on the grid at ``sigma**2 * 2**-53``.
    Remainders of large terms land exactly on the second grid, so values far below
    cancelling extremes survive in the final remainders.
    """
    first = (value + sigma) - sigma
    remainder = value - first
    second_sigma = sigma * sigma * 2.0**-53
    second = (remainder + second_sigma) - second_sigma
    return first, second, remainder - second


def _compensated(value, sigma):
    """Compensated sum and the magnitude of the remainders it adds in any order."""
    first, second, remainder = _split_terms(value, sigma)
    return (first.sum() + second.sum()) + remainder.sum(), remainder.abs().sum()


def _resolved(count, remainder_magnitude, scaled_mean):
    """Whether the summation error bound is negligible beside the resulting mean.

    Adding ``count`` remainders in any order errs by at most ``count * 2**-53`` of
    their magnitude; with a factor two for combining the parts, that bound must
    stay within ``2**-36`` of ``count`` times the mean, below the compact-versus-unit
    parity tolerance. Ordinary rounding passes; remainders that cancel far above
    the mean, which the split cannot resolve exactly, do not.
    """
    return count * remainder_magnitude * _SUM_ERROR <= _MEAN_TOLERANCE * scaled_mean.abs()


def _median_overflows(centered, exponent):
    """Whether a normalized difference would overflow at the original scale.

    Normalized magnitudes stay below two, so only the two largest exponents can overflow.
    """
    magnitude = centered.abs()
    return ((exponent == 1024) & (magnitude >= 1.0)) | ((exponent == 1023) & (magnitude >= 2.0))


def _component_exponent_parts(maximum, sum_squares):
    """Scale by the arm's L2 norm while keeping its maximum SQL-local.

    With ``m=max(abs(v))`` and ``q=sum((v/m)**2)``, the norm is ``m*sqrt(q)``.
    Writing ``m=s*2**e`` with ``e=floor(log2(m))+1`` lets us calculate its
    exponent as ``e+floor(log2(s*sqrt(q)))+1`` without materializing an
    overflowing norm. Since ``q`` terms are at most one, every normalized
    outcome is at most one and offsets at most two; the compensated-sum and
    ``2**-36`` range checks therefore keep their existing bounds.
    """
    maximum = cast(ir.FloatingValue, maximum.cast("float64").fill_null(0.0))
    valid = (cast(Any, maximum) > 0) & ~maximum.isnan() & ~maximum.isinf()
    max_exponent = ibis.ifelse(valid, (maximum.log2().floor() + 1).clip(upper=1024), 0).cast(
        "int64"
    )
    scaled_maximum = _binary_normalize(maximum, max_exponent)
    scaled_norm = scaled_maximum * sum_squares.sqrt()
    return max_exponent, ibis.ifelse(valid, scaled_norm, 1.0), valid


def _join_on(left, right, keys):
    joined = left.join(right, [left[name].identical_to(right[name]) for name in keys])
    return joined.select(
        *[left[name] for name in left.columns],
        **{name: right[name] for name in right.columns if name not in keys},
    )


def _cell_relation(relation, names, outcomes):
    keys = ("group_id", *names)

    maxima = relation.group_by("group_id").aggregate(
        **{
            f"_max_{f}": relation[name].cast("float64").abs().max()
            for f, name in enumerate(outcomes)
        }
    )
    relation = _join_on(relation, maxima, ("group_id",))
    norm_stats = relation.group_by("group_id").aggregate(
        _arm_count=relation.count(),
        **{
            name: expression
            for f, outcome in enumerate(outcomes)
            for name, expression in (
                (f"_norm_max_{f}", relation[f"_max_{f}"].max()),
                (
                    f"_sum_squares_{f}",
                    (
                        relation[outcome].cast("float64")
                        / ibis.ifelse(relation[f"_max_{f}"] > 0, relation[f"_max_{f}"], 1.0)
                    )
                    .pow(2)
                    .sum(),
                ),
            )
        },
    )
    norm_stats = norm_stats.mutate(
        **{
            name: value
            for f in range(len(outcomes))
            for parts in (
                _component_exponent_parts(
                    norm_stats[f"_norm_max_{f}"], norm_stats[f"_sum_squares_{f}"]
                ),
            )
            for name, value in (
                (f"_max_exponent_{f}", parts[0]),
                (f"_scaled_norm_{f}", parts[1]),
                (f"_norm_valid_{f}", parts[2]),
            )
        }
    )
    exponents = norm_stats.mutate(
        **{
            f"exponent_{f}": ibis.ifelse(
                norm_stats[f"_norm_valid_{f}"],
                (
                    norm_stats[f"_max_exponent_{f}"]
                    + norm_stats[f"_scaled_norm_{f}"].log2().floor()
                    + 1
                ).clip(upper=1024),
                0,
            ).cast("int64")
            for f in range(len(outcomes))
        }
    ).select(
        "group_id",
        "_arm_count",
        *(f"exponent_{f}" for f in range(len(outcomes))),
    )
    relation = _join_on(relation, exponents, ("group_id",))
    relation = relation.mutate(
        _sigma=2.0 ** (relation["_arm_count"].cast("float64").log2().ceil() + 2),
        **{
            f"_rank_{f}": ibis.row_number().over(group_by="group_id", order_by=relation[name])
            for f, name in enumerate(outcomes)
        },
    )
    # The per-unit fitter's offset: the upper median, kept inside the warehouse.
    medians = relation.group_by("group_id").aggregate(
        **{
            f"_median_{f}": relation[name]
            .cast("float64")
            .max(where=relation[f"_rank_{f}"] == relation["_arm_count"] // 2)
            for f, name in enumerate(outcomes)
        }
    )
    relation = _join_on(relation, medians, ("group_id",))
    overflow = relation.group_by("group_id").aggregate(
        **{
            f"_overflow_{f}": _median_overflows(
                _binary_normalize(relation[name], relation[f"exponent_{f}"])
                - _binary_normalize(relation[f"_median_{f}"], relation[f"exponent_{f}"]),
                relation[f"exponent_{f}"],
            ).any()
            for f, name in enumerate(outcomes)
        }
    )
    relation = _join_on(relation, overflow, ("group_id",))
    relation = relation.mutate(
        **{
            f"_base_{f}": ibis.ifelse(relation[f"_overflow_{f}"], 0.0, relation[f"_median_{f}"])
            for f in range(len(outcomes))
        }
    )
    shifted = relation.mutate(
        **{
            f"_base_norm_{f}": _binary_normalize(relation[f"_base_{f}"], relation[f"exponent_{f}"])
            for f in range(len(outcomes))
        }
    )
    shifted = shifted.mutate(
        **{
            f"_offset_{f}": _binary_normalize(shifted[name], shifted[f"exponent_{f}"])
            - shifted[f"_base_norm_{f}"]
            for f, name in enumerate(outcomes)
        }
    )
    arm_sums = {
        f: _compensated(shifted[f"_offset_{f}"], shifted["_sigma"]) for f in range(len(outcomes))
    }
    totals = shifted.group_by("group_id").aggregate(
        **{f"_arm_sum_{f}": total for f, (total, _) in arm_sums.items()},
        **{f"_arm_remainder_{f}": remainder for f, (_, remainder) in arm_sums.items()},
    )
    joined = _join_on(shifted, totals, ("group_id",))
    # Transferred references are arm means, never a selected unit's outcome.
    referenced = joined.mutate(
        **{
            f"ref_{f}": joined[f"_base_{f}"]
            + _binary_denormalize(
                joined[f"_arm_sum_{f}"] / joined["_arm_count"], joined[f"exponent_{f}"]
            )
            for f in range(len(outcomes))
        }
    )
    centered = referenced.mutate(
        **{
            f"_shift_{f}": _binary_normalize(referenced[f"ref_{f}"], referenced[f"exponent_{f}"])
            - referenced[f"_base_norm_{f}"]
            for f in range(len(outcomes))
        }
    )
    aggregates = {"n": centered.count()}
    for f, name in enumerate(outcomes):
        value = centered[name].cast("float64")
        finite = value.notnull() & ~value.isnan() & ~value.isinf()
        total, remainder = _compensated(centered[f"_offset_{f}"], centered["_sigma"])
        count = centered.count()
        arm_count = centered["_arm_count"].min()
        base = centered[f"_base_norm_{f}"].min()
        aggregates.update(
            {
                f"ref_{f}": centered[f"ref_{f}"].min(),
                f"exponent_{f}": centered[f"exponent_{f}"].min(),
                f"_sum_{f}": total,
                f"first_{f}": total - count * centered[f"_shift_{f}"].min(),
                f"_precise_{f}": _resolved(count, remainder, count * base + total)
                & _resolved(
                    arm_count,
                    centered[f"_arm_remainder_{f}"].min(),
                    arm_count * base + centered[f"_arm_sum_{f}"].min(),
                ),
                f"contributors_{f}": finite.ifelse(1, 0).cast("int64").sum(),
                f"distinct_{f}": value.min() < value.max(),
                f"max_centered_{f}": centered[f"_offset_{f}"].abs().max(),
            }
        )
    first = centered.group_by(list(keys)).aggregate(**aggregates)
    means = first.select(
        *keys, **{f"_mean_{f}": first[f"_sum_{f}"] / first.n for f in range(len(outcomes))}
    )
    residuals = _join_on(centered, means, keys)
    residuals = residuals.mutate(
        **{
            f"_residual_{f}": residuals[f"_offset_{f}"] - residuals[f"_mean_{f}"]
            for f in range(len(outcomes))
        }
    )
    cross = residuals.group_by(list(keys)).aggregate(
        **{
            f"cross_{f}_{g}": (residuals[f"_residual_{f}"] * residuals[f"_residual_{g}"]).sum()
            for f in range(len(outcomes))
            for g in range(f, len(outcomes))
        }
    )
    return _join_on(first, cross, keys)


def _preflight(cells, names, components, request, metric):
    positive, contributors = [], []
    for f in range(components):
        scatter = cells[f"cross_{f}_{f}"]
        error = 64 * np.finfo(float).eps * cells.n * cells[f"max_centered_{f}"] ** 2
        positive.append(
            cells[f"distinct_{f}"] & (scatter > error) & ~scatter.isnan() & ~scatter.isinf()
        )
        contributors.append(cells[f"contributors_{f}"])
    label_bytes = ibis.literal(0, type="int64")
    for name in ("group_id", *names):
        label_bytes += 12 * cells[name].cast("string").length().fill_null(0) + 16
    # Labels occur in both patterns and level metadata; dense arrays include zero-count cells.
    cell_count = cells.count()
    arms = cells.group_id.nunique()
    schema_bytes = sum(12 * len(name) + 64 for name in names)
    dense_bytes = 32 + 32 * (components + components * components)
    identity_bytes = len(
        json.dumps(
            {
                "identity": request.identity,
                "method": request.method.model_dump(mode="json"),
                "metric": metric.model_dump(mode="json"),
            },
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )
    encoded_bytes = (
        4096
        + identity_bytes
        + cell_count * (512 + schema_bytes)
        + 2 * label_bytes.sum()
        + arms * cell_count * dense_bytes
    )
    return cells.aggregate(
        cell_count=cell_count,
        unit_count=cells.n.sum(),
        min_contributors=cast(ir.IntegerColumn, ibis.least(*contributors)).min(),
        components_complete=cast(
            ir.BooleanColumn, ibis.and_(*[count == cells.n for count in contributors])
        ).all(),
        scatter_positive=cast(ir.BooleanColumn, ibis.and_(*positive)).all(),
        encoded_bytes=encoded_bytes,
        **{
            f"range_arm_{f}": cells.group_id.min(where=~cells[f"_precise_{f}"])
            for f in range(components)
        },
    )


def _rows(value):
    return list(nw.from_native(value, eager_only=True).iter_rows(named=True))


def _scratch_columns(components):
    columns = {"_arm_count", "_sigma"}
    for f in range(components):
        columns.update(
            {
                f"_max_{f}",
                f"exponent_{f}",
                f"_rank_{f}",
                f"_median_{f}",
                f"_overflow_{f}",
                f"_base_{f}",
                f"_base_norm_{f}",
                f"_centered_{f}",
                f"_offset_{f}",
                f"_arm_sum_{f}",
                f"_arm_remainder_{f}",
                f"ref_{f}",
                f"_shift_{f}",
                f"first_{f}",
                f"_precise_{f}",
                f"contributors_{f}",
                f"distinct_{f}",
                f"max_centered_{f}",
                f"_sum_{f}",
                f"_mean_{f}",
                f"_residual_{f}",
            }
        )
        for g in range(f, components):
            columns.add(f"cross_{f}_{g}")
    return columns


def _rename_scratch_inputs(relation, names, components):
    scratch = _scratch_columns(components)
    occupied = set(relation.columns)
    rename = {}
    safe_names = {}
    for index, name in enumerate(sorted(occupied & scratch)):
        candidate = f"__increment_cuped_input_{index}"
        while candidate in occupied or candidate in scratch:
            candidate += "_"
        rename[candidate] = name
        safe_names[name] = candidate
        occupied.add(candidate)
    return relation.rename(rename), tuple(safe_names.get(name, name) for name in names)


def categorical_cells_from_relation(
    relation, execute, *, metric, request: _ResolvedCupedRequest, magnitude_checked=False
) -> CategoricalCells:
    declarations = request.method.covariates or ()
    if any(item.kind != "categorical" for item in declarations):
        general_qr_required(metric, "continuous_covariates")
    names = tuple(item.name for item in declarations)
    outcomes = ("y", "y_den") if metric.type == "ratio" else ("y",)
    if _known_outcome_domain(metric, relation, outcomes):
        general_qr_required(metric, "outcome_domain")
    relation, query_names = _rename_scratch_inputs(relation, names, len(outcomes))
    relation = relation.filter(relation.group_id.isin(request.arm_order))
    if not magnitude_checked:
        require_warehouse_magnitude_range(
            relation,
            metric=metric.name,
            components=tuple(
                (label, outcome)
                for label, outcome in zip(
                    ("numerator", "denominator") if metric.type == "ratio" else ("outcome",),
                    outcomes,
                    strict=True,
                )
            ),
            execute=execute,
        )
    cells = _cell_relation(relation, query_names, outcomes)
    preflight = _rows(execute(_preflight(cells, query_names, len(outcomes), request, metric)))[0]
    count = int(preflight["cell_count"])
    if not 0 < count <= MAX_CELLS:
        general_qr_required(metric, "cell_count")
    if preflight["unit_count"] < MIN_COMPRESSION * count:
        general_qr_required(metric, "compression")
    if not preflight["components_complete"] or preflight["min_contributors"] < MIN_CONTRIBUTORS:
        general_qr_required(metric, "component_contributors")
    if not preflight["scatter_positive"]:
        general_qr_required(metric, "component_scatter")
    labels = ("numerator", "denominator") if metric.type == "ratio" else ("outcome",)
    for f, label in enumerate(labels):
        if preflight[f"range_arm_{f}"] is not None:
            general_qr_required(
                metric, "component_range", arm=preflight[f"range_arm_{f}"], component=label
            )
    if preflight["encoded_bytes"] is None or preflight["encoded_bytes"] > MAX_ENCODED_BYTES:
        general_qr_required(metric, "encoded_bytes")
    fields = [*query_names, "group_id", "n"]
    fields += [f"{kind}_{f}" for f in range(len(outcomes)) for kind in ("ref", "first", "exponent")]
    fields += [f"cross_{f}_{g}" for f in range(len(outcomes)) for g in range(f, len(outcomes))]
    rows = _rows(execute(cells.select(*fields)))
    return _as_cells(rows, query_names, len(outcomes), request, covariate_names=names)


def _as_cells(rows, names, components, request, *, covariate_names=None):
    covariate_names = names if covariate_names is None else covariate_names
    patterns = tuple(sorted({tuple(row[name] for name in names) for row in rows}, key=_pattern_key))
    lookup = {pattern: k for k, pattern in enumerate(patterns)}
    arms = request.arm_order
    counts = np.zeros((len(arms), len(patterns)), dtype=np.int64)
    references = np.zeros((components, len(arms)))
    exponents = [[0] * len(arms) for _ in range(components)]
    first = np.zeros((len(arms), components, len(patterns)))
    cross = np.zeros((len(arms), components, components, len(patterns)))
    for row in rows:
        a = arms.index(row["group_id"])
        k = lookup[tuple(row[name] for name in names)]
        counts[a, k] = row["n"]
        for f in range(components):
            references[f, a] = row[f"ref_{f}"]
            exponents[f][a] = int(row[f"exponent_{f}"])
            first[a, f, k] = row[f"first_{f}"]
            for g in range(f, components):
                cross[a, f, g, k] = cross[a, g, f, k] = row[f"cross_{f}_{g}"]
    levels = tuple(
        tuple(
            sorted(
                {pattern[column] for pattern in patterns},
                key=lambda value: (value is None, value or ""),
            )
        )
        for column in range(len(names))
    )
    return CategoricalCells(
        identity=request.identity,
        arms=arms,
        patterns=patterns,
        counts=_freeze_payload(counts),
        levels=levels,
        null_metadata=tuple(
            (item.missing, None in level)
            for item, level in zip(request.method.covariates or (), levels, strict=True)
        ),
        outcome_references=_freeze_payload(references),
        centered_first_moments=_freeze_payload(first),
        symmetric_cross_moments=_freeze_payload(cross),
        covariate_names=covariate_names,
        outcome_exponents=_freeze_payload(exponents),
    )
