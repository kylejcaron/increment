"""Warehouse ingress for the overlapping-experiments family.

Validation and per-cell moments run as Ibis aggregates, so only counts,
bounded examples, and four cell summaries per metric leave the warehouse.
The validation counts feed the same refusal sequence as the dataframe
ingress, and the moments feed the same estimator.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

import ibis
import ibis.expr.datatypes as dtypes
import ibis.expr.types as ir

from increment._overlap.admission import (
    Admission,
    CellMoments,
    EligibilityAudit,
    cell_arms,
    check_bindings,
)
from increment._overlap.checks import (
    ExperimentViolations,
    TargetViolations,
    Violations,
    example_labels,
    refuse_empty_target,
    refuse_target_violations,
    refuse_unit_violations,
)
from increment._overlap.contrasts import CELLS
from increment._overlap.design import ExperimentColumns, OverlapDesign, OverlapExperiment
from increment._overlap.refusals import _raise

_EXAMPLES = 3


def _one(expr: ir.Table) -> dict[str, Any]:
    return expr.to_pyarrow().to_pylist()[0]


def _int(value: object) -> int:
    """A count from an aggregate row; SQL ``SUM`` over zero rows is null."""
    if value is None:
        return 0
    if not isinstance(value, int):
        raise AssertionError(f"expected an integer count, got {value!r}")
    return value


def _experiment_violations(
    table: ir.Table, experiment: OverlapExperiment, cols: ExperimentColumns
) -> ExperimentViolations:
    eligibility_missing = _int(_one(table.aggregate(n=table[cols.eligible].isnull().sum()))["n"])
    if eligibility_missing:
        return ExperimentViolations(eligibility_missing=eligibility_missing)
    eligible = table[cols.eligible].cast("boolean")
    arm = table[cols.assignment].cast("string")
    assigned = arm.notnull()
    unknown = assigned & ~arm.isin([experiment.control, experiment.treatment])
    counts = _one(
        table.aggregate(
            unknown=unknown.sum(),
            eligible_unassigned=(eligible & ~assigned).sum(),
            ineligible_assigned=(~eligible & assigned).sum(),
            assigned_at_missing=(assigned & table[cols.assigned_at].isnull()).sum(),
        )
    )
    examples: tuple[str, ...] = ()
    if _int(counts["unknown"]):
        rows = (
            table.filter(unknown)
            .select(arm=arm)
            .distinct()
            .order_by("arm")
            .limit(_EXAMPLES)
            .to_pyarrow()
            .to_pylist()
        )
        examples = example_labels(r["arm"] for r in rows)
    return ExperimentViolations(
        unknown_arm=_int(counts["unknown"]),
        unknown_arm_examples=examples,
        eligible_unassigned=_int(counts["eligible_unassigned"]),
        ineligible_assigned=_int(counts["ineligible_assigned"]),
        assigned_at_missing=_int(counts["assigned_at_missing"]),
    )


def _unit_violations(
    table: ir.Table,
    design: OverlapDesign,
    unit: str,
    first: ExperimentColumns,
    second: ExperimentColumns,
) -> Violations:
    missing = _int(_one(table.aggregate(n=table[unit].isnull().sum()))["n"])
    per_unit = table.filter(table[unit].notnull()).group_by(unit).aggregate(rows=ibis._.count())
    duplicated = per_unit.filter(per_unit.rows > 1)
    count = _int(_one(duplicated.aggregate(n=ibis._.count()))["n"])
    examples: tuple[str, ...] = ()
    if count:
        rows = duplicated.order_by(unit).limit(_EXAMPLES).to_pyarrow().to_pylist()
        examples = example_labels(r[unit] for r in rows)
    return Violations(
        missing_units=missing,
        duplicate_units=count,
        duplicate_examples=examples,
        first=_experiment_violations(table, design.first, first),
        second=_experiment_violations(table, design.second, second),
    )


def _timestamps(table: ir.Table, names: tuple[str, ...]) -> ir.Table:
    zones: set[bool] = set()
    casts = {}
    for name in names:
        dtype = table.schema()[name]
        if dtype.is_date():
            casts[name] = table[name].cast(dtypes.Timestamp(scale=6))
            zones.add(False)
        elif isinstance(dtype, dtypes.Timestamp):
            casts[name] = table[name].cast(dtypes.Timestamp(timezone=dtype.timezone, scale=6))
            zones.add(dtype.timezone is not None)
        else:
            _raise("overlap.timing.dtype", column=name, dtype=str(dtype))
    if len(zones) > 1:
        _raise("overlap.timing.awareness_mixed", columns=names)
    return table.mutate(**casts)


def _target_violations(
    target: ir.Table,
    first: ExperimentColumns,
    second: ExperimentColumns,
    outcome_start: str,
    outcome_end: str,
    metrics: Mapping[str, str],
) -> TargetViolations:
    start, end = target[outcome_start], target[outcome_end]
    latest = ibis.greatest(target[first.assigned_at], target[second.assigned_at])
    timing = _one(
        target.aggregate(
            window_missing=(start.isnull() | end.isnull()).sum(),
            window_order=(end <= start).sum(),
            precedes=(start < latest).sum(),
        )
    )
    if _int(timing["window_missing"]):
        return TargetViolations(window_missing=_int(timing["window_missing"]))
    if _int(timing["window_order"]):
        return TargetViolations(window_order=_int(timing["window_order"]))
    if _int(timing["precedes"]):
        return TargetViolations(outcome_precedes_assignment=_int(timing["precedes"]))
    lengths = (
        target.select(length=end.delta(start, unit="microsecond")).distinct().order_by("length")
    )
    distinct = _int(_one(lengths.aggregate(n=ibis._.count()))["n"])
    if distinct > 1:
        shown = lengths.limit(_EXAMPLES).to_pyarrow().to_pylist()
        return TargetViolations(
            distinct_window_lengths=distinct,
            window_lengths=tuple(str(dt.timedelta(microseconds=r["length"])) for r in shown),
        )
    missing, nonfinite = [], []
    for metric, column in metrics.items():
        dtype = target.schema()[column]
        if not dtype.is_numeric():
            _raise("overlap.outcome.non_numeric", metric=metric, column=column, dtype=str(dtype))
        col = target[column]
        aggregates = {"missing": col.isnull().sum()}
        if dtype.is_floating():
            aggregates["nonfinite"] = (col.isnan() | col.isinf()).sum()
        counts = _one(target.aggregate(**aggregates))
        missing.append((metric, _int(counts["missing"])))
        nonfinite.append((metric, _int(counts.get("nonfinite"))))
    return TargetViolations(metric_missing=tuple(missing), metric_nonfinite=tuple(nonfinite))


def admit_table(
    table: ir.Table,
    design: OverlapDesign,
    *,
    unit: str,
    columns: Mapping[str, ExperimentColumns],
    outcome_start: str,
    outcome_end: str,
    metrics: Mapping[str, str],
) -> Admission:
    first_cols, second_cols = check_bindings(design, columns, metrics)
    required = (
        unit,
        *first_cols.names,
        *second_cols.names,
        outcome_start,
        outcome_end,
        *metrics.values(),
    )
    missing = tuple(dict.fromkeys(c for c in required if c not in table.columns))
    if missing:
        _raise("overlap.ingress.missing_columns", missing=missing)
    refuse_unit_violations(design, _unit_violations(table, design, unit, first_cols, second_cols))
    table = _timestamps(
        table, (first_cols.assigned_at, second_cols.assigned_at, outcome_start, outcome_end)
    )
    e1 = table[first_cols.eligible].cast("boolean")
    e2 = table[second_cols.eligible].cast("boolean")
    counts = _one(
        table.aggregate(
            units=ibis._.count(),
            jointly=(e1 & e2).sum(),
            first_only=(e1 & ~e2).sum(),
            second_only=(~e1 & e2).sum(),
            neither=(~e1 & ~e2).sum(),
        )
    )
    audit = EligibilityAudit(
        units=_int(counts["units"]),
        jointly_eligible=_int(counts["jointly"]),
        first_only_eligible=_int(counts["first_only"]),
        second_only_eligible=_int(counts["second_only"]),
        neither_eligible=_int(counts["neither"]),
    )
    refuse_empty_target(design, audit.jointly_eligible)
    target = table.filter(e1 & e2)
    refuse_target_violations(
        _target_violations(target, first_cols, second_cols, outcome_start, outcome_end, metrics)
    )
    window = _one(
        target.limit(1).select(
            length=target[outcome_end].delta(target[outcome_start], unit="microsecond")
        )
    )["length"]
    keyed = target.mutate(
        __first=(target[first_cols.assignment].cast("string") == design.first.treatment).cast(
            "int8"
        ),
        __second=(target[second_cols.assignment].cast("string") == design.second.treatment).cast(
            "int8"
        ),
    )
    aggregates: dict[str, Any] = {"__n": ibis._.count()}
    for i, column in enumerate(metrics.values()):
        value = keyed[column].cast("float64")
        aggregates[f"__mean_{i}"] = value.mean()
        aggregates[f"__var_{i}"] = value.var(how="sample")
    grouped = {
        (r["__first"], r["__second"]): r
        for r in keyed.group_by(["__first", "__second"])
        .aggregate(**aggregates)
        .to_pyarrow()
        .to_pylist()
    }
    moments: list[CellMoments] = []
    for i, metric in enumerate(metrics):
        for cell in CELLS:
            row = grouped.get(cell)
            n = _int(row["__n"]) if row else 0
            first_arm, second_arm = cell_arms(design, cell)
            moments.append(
                CellMoments(
                    metric=metric,
                    cell=cell,
                    first_arm=first_arm,
                    second_arm=second_arm,
                    n=n,
                    mean=float(row[f"__mean_{i}"]) if row and n else None,
                    variance=float(row[f"__var_{i}"]) if row and n >= 2 else None,
                )
            )
    return Admission(
        design=design,
        metrics=tuple(metrics),
        moments=tuple(moments),
        audit=audit,
        outcome_window=dt.timedelta(microseconds=int(window)),
    )
