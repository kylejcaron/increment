"""Dataframe ingress: narwhals validation and per-cell moments for one row per unit.

Pure narwhals over pandas, polars, or pyarrow; never imports Ibis or ``increment.query``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

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


def _count(mask: nw.Series) -> int:
    return int(mask.sum())


def _experiment_violations(
    frame: nw.DataFrame, experiment: OverlapExperiment, cols: ExperimentColumns
) -> ExperimentViolations:
    eligible = frame.get_column(cols.eligible)
    assignment = frame.get_column(cols.assignment)
    assigned_at = frame.get_column(cols.assigned_at)
    eligibility_missing = _count(eligible.is_null())
    if eligibility_missing:
        return ExperimentViolations(eligibility_missing=eligibility_missing)
    eligible = eligible.cast(nw.Boolean)
    assigned = ~assignment.is_null()
    arms = assignment.cast(nw.String)
    unknown = assigned & ~arms.is_in([experiment.control, experiment.treatment])
    return ExperimentViolations(
        unknown_arm=_count(unknown),
        unknown_arm_examples=example_labels(arms.filter(unknown).to_list()),
        eligible_unassigned=_count(eligible & ~assigned),
        ineligible_assigned=_count(~eligible & assigned),
        assigned_at_missing=_count(assigned & assigned_at.is_null()),
    )


def _timestamps(frame: nw.DataFrame, names: tuple[str, ...]) -> nw.DataFrame:
    """Cast date columns to timestamps and refuse mixed timezone awareness."""
    zones: set[bool] = set()
    exprs = []
    for name in names:
        dtype = frame.schema[name]
        if dtype == nw.Date:
            exprs.append(nw.col(name).cast(nw.Datetime("us")))
            zones.add(False)
        elif isinstance(dtype, nw.Datetime):
            exprs.append(nw.col(name).cast(nw.Datetime("us", dtype.time_zone)))
            zones.add(dtype.time_zone is not None)
        else:
            _raise("overlap.timing.dtype", column=name, dtype=str(dtype))
    if len(zones) > 1:
        _raise("overlap.timing.awareness_mixed", columns=names)
    return frame.with_columns(exprs)


def _target_violations(
    target: nw.DataFrame,
    first: ExperimentColumns,
    second: ExperimentColumns,
    outcome_start: str,
    outcome_end: str,
    metrics: Mapping[str, str],
) -> TargetViolations:
    start, end = target.get_column(outcome_start), target.get_column(outcome_end)
    window_missing = _count(start.is_null() | end.is_null())
    if window_missing:
        return TargetViolations(window_missing=window_missing)
    window_order = _count(end <= start)
    if window_order:
        return TargetViolations(window_order=window_order)
    latest = target.select(
        nw.max_horizontal(nw.col(first.assigned_at), nw.col(second.assigned_at)).alias("latest")
    ).get_column("latest")
    precedes = _count(start < latest)
    if precedes:
        return TargetViolations(outcome_precedes_assignment=precedes)
    lengths = (end - start).dt.total_microseconds().unique().sort().to_list()
    if len(lengths) > 1:
        return TargetViolations(
            distinct_window_lengths=len(lengths),
            window_lengths=tuple(str(dt.timedelta(microseconds=v)) for v in lengths[:3]),
        )
    missing, nonfinite = [], []
    for metric, column in metrics.items():
        dtype = target.schema[column]
        if not dtype.is_numeric():
            _raise("overlap.outcome.non_numeric", metric=metric, column=column, dtype=str(dtype))
        values = target.get_column(column)
        missing.append((metric, _count(values.is_null())))
        nonfinite.append((metric, _count(~values.cast(nw.Float64).is_finite() & ~values.is_null())))
    return TargetViolations(metric_missing=tuple(missing), metric_nonfinite=tuple(nonfinite))


def cell_moments(
    design: OverlapDesign,
    metric: str,
    first_treated: np.ndarray,
    second_treated: np.ndarray,
    values: np.ndarray,
) -> tuple[CellMoments, ...]:
    """Per-cell mean and sample variance; nulls below the identifying count."""
    out = []
    for cell in CELLS:
        y = values[(first_treated == cell[0]) & (second_treated == cell[1])]
        n = int(y.size)
        mean = variance = None
        if n:
            # Centering on the midpoint bounds every deviation by half the cell's range,
            # so the mean of values near the float limit does not overflow.
            center = float(y.min()) / 2.0 + float(y.max()) / 2.0
            deviation = y - center
            offset = float((deviation / n).sum())
            mean = center + offset
            if n >= 2:
                resid = deviation - offset
                scale = float(np.abs(resid).max())
                variance = (
                    scale * scale * float(((resid / scale) ** 2).sum()) / (n - 1)
                    if scale > 0.0
                    else 0.0
                )
        first_arm, second_arm = cell_arms(design, cell)
        out.append(
            CellMoments(
                metric=metric,
                cell=cell,
                first_arm=first_arm,
                second_arm=second_arm,
                n=n,
                mean=mean,
                variance=variance,
            )
        )
    return tuple(out)


def admit_frame(
    frame: IntoDataFrame,
    design: OverlapDesign,
    *,
    unit: str,
    columns: Mapping[str, ExperimentColumns],
    outcome_start: str,
    outcome_end: str,
    metrics: Mapping[str, str],
) -> Admission:
    first_cols, second_cols = check_bindings(design, columns, metrics)
    nwf = nw.from_native(frame, eager_only=True)
    required = (
        unit,
        *first_cols.names,
        *second_cols.names,
        outcome_start,
        outcome_end,
        *metrics.values(),
    )
    missing = tuple(dict.fromkeys(c for c in required if c not in nwf.columns))
    if missing:
        _raise("overlap.ingress.missing_columns", missing=missing)
    units = nwf.get_column(unit)
    duplicated = units.is_duplicated() & ~units.is_null()
    refuse_unit_violations(
        design,
        Violations(
            missing_units=_count(units.is_null()),
            duplicate_units=int(units.filter(duplicated).n_unique()),
            duplicate_examples=example_labels(units.filter(duplicated).to_list()),
            first=_experiment_violations(nwf, design.first, first_cols),
            second=_experiment_violations(nwf, design.second, second_cols),
        ),
    )
    nwf = _timestamps(
        nwf, (first_cols.assigned_at, second_cols.assigned_at, outcome_start, outcome_end)
    )
    e1 = nwf.get_column(first_cols.eligible).cast(nw.Boolean)
    e2 = nwf.get_column(second_cols.eligible).cast(nw.Boolean)
    audit = EligibilityAudit(
        units=len(nwf),
        jointly_eligible=_count(e1 & e2),
        first_only_eligible=_count(e1 & ~e2),
        second_only_eligible=_count(~e1 & e2),
        neither_eligible=_count(~e1 & ~e2),
    )
    refuse_empty_target(design, audit.jointly_eligible)
    target = nwf.filter(e1 & e2)
    refuse_target_violations(
        _target_violations(target, first_cols, second_cols, outcome_start, outcome_end, metrics)
    )
    window = dt.timedelta(
        microseconds=int(
            (target.get_column(outcome_end) - target.get_column(outcome_start))
            .dt.total_microseconds()
            .to_list()[0]
        )
    )
    first_treated = (
        (target.get_column(first_cols.assignment).cast(nw.String) == design.first.treatment)
        .to_numpy()
        .astype(np.int8)
    )
    second_treated = (
        (target.get_column(second_cols.assignment).cast(nw.String) == design.second.treatment)
        .to_numpy()
        .astype(np.int8)
    )
    moments: list[CellMoments] = []
    for metric, column in metrics.items():
        values = target.get_column(column).cast(nw.Float64).to_numpy()
        moments.extend(cell_moments(design, metric, first_treated, second_treated, values))
    return Admission(
        design=design,
        metrics=tuple(metrics),
        moments=tuple(moments),
        audit=audit,
        outcome_window=window,
    )
