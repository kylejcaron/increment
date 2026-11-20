"""Capture bounded overlap evidence from four local, typed snapshot frames."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime
from functools import reduce
from math import isfinite
from operator import or_
from typing import TYPE_CHECKING, Any, NoReturn, cast

import narwhals as nw

from increment._assignment_refusals import refuse_assignment_invalid
from increment._overlap_evidence import (
    CELL_IDS,
    DECISION_STATUSES,
    DELIVERY_KEYS,
    FACTOR_IDS,
    GATE_KEYS,
    MAX_EXACT_INTEGER,
    OVERLAP_RELATIONS,
    PREASSIGNED_GATE_COLUMNS,
    CellId,
    DeliveryCellId,
    FeedRole,
    GateDisposition,
    OverlapCell,
    OverlapColumn,
    OverlapDeliveryCount,
    _utc_microsecond,
    overlap_binary_moments,
)
from increment._overlap_refusals import CAPTURE_INVALID, EVIDENCE_INVALID, NUMERIC_RANGE
from increment.errors import refuse
from increment.semantics.overlap import OverlapCapture, OverlapEvidence, OverlapFactor, OverlapPlan

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame


# Narwhals row values are dynamically typed by the native frame. Validate them
# against the physical column descriptors before joining or reducing rows.
type _Row = dict[str, Any]
type _Key = tuple[str, ...]
_SURROGATE = re.compile(r"[\ud800-\udfff]")


def _require(role: FeedRole, reason: str, count: int) -> None:
    if count:
        refuse(EVIDENCE_INVALID, role=role, reason=reason, count=count)


def _physical(dtype: nw.dtypes.DType, kind: str) -> bool:
    match kind:
        case "string":
            return dtype == nw.String
        case "boolean":
            return dtype == nw.Boolean
        case "integer":
            return dtype.is_integer()
        case "float":
            return dtype.is_float()
        case "timestamp":
            return isinstance(dtype, nw.Datetime) and dtype.time_zone is not None
    return False


def _frame(
    source: IntoDataFrame,
    role: FeedRole,
    columns: tuple[OverlapColumn, ...],
) -> tuple[nw.DataFrame[Any], tuple[str, ...]]:
    try:
        frame = nw.from_native(source, eager_only=True)
        schema = frame.schema
    except Exception:
        # Backend conversion errors can contain source data or implementation details.
        pass
    else:
        _require(role, "missing_columns", sum(column.name not in schema for column in columns))
        _require(
            role,
            "invalid_column_type",
            sum(not _physical(schema[column.name], column.kind) for column in columns),
        )
        nanos = tuple(
            column.name
            for column in columns
            if isinstance(dtype := schema[column.name], nw.Datetime) and dtype.time_unit == "ns"
        )
        return frame, nanos
    refuse(EVIDENCE_INVALID, role=role, reason="unreadable_frame", count=1)


def _value_ok(value: object, column: OverlapColumn) -> bool:
    if value is None:
        return column.nullable
    match column.kind:
        case "string":
            return type(value) is str and bool(value) and _SURROGATE.search(value) is None
        case "boolean":
            return type(value) is bool
        case "integer":
            return type(value) is int
        case "float":
            return type(value) is float
        case "timestamp":
            return isinstance(value, datetime)
    return False


def _valid_row(row: _Row, columns: tuple[OverlapColumn, ...]) -> bool:
    if row.pop("__overlap_submicrosecond", False):
        return False
    try:
        for column in columns:
            value = row[column.name]
            if not _value_ok(value, column):
                return False
            if value is not None and column.kind == "timestamp":
                row[column.name] = _utc_microsecond(value)
    except ValueError:
        return False
    return True


def _rows(
    frame: nw.DataFrame[Any],
    role: FeedRole,
    columns: tuple[OverlapColumn, ...],
    nanos: tuple[str, ...],
    plan: OverlapPlan,
) -> list[_Row]:
    predicate = nw.col("study_id") == plan.study_id
    if role == "decisions":
        scope = nw.col("target")
        predicate &= scope.is_null() | scope.is_in(("", plan.a.target, plan.b.target))
    elif role == "deliveries":
        scope = nw.col("experiment")
        predicate &= scope.is_null() | scope.is_in(("", plan.a.experiment, plan.b.experiment))
    elif role == "outcomes":
        scope = nw.col("metric")
        predicate &= scope.is_null() | scope.is_in(("", *(metric.name for metric in plan.metrics)))
    try:
        scoped = frame.filter(predicate).select(*(column.name for column in columns))
        if nanos:
            bad = reduce(
                or_, ((nw.col(name).dt.nanosecond() % 1000 != 0).fill_null(False) for name in nanos)
            )
            scoped = scoped.with_columns(bad.alias("__overlap_submicrosecond"))
        if scoped.implementation.is_pandas_like():
            # Keep the physical null bitmap without treating the index as a column.
            rows = nw.maybe_reset_index(scoped).to_arrow().to_pylist()
        else:
            rows = list(scoped.iter_rows(named=True))
    except Exception:
        pass
    else:
        _require(role, "invalid_value", sum(not _valid_row(row, columns) for row in rows))
        return rows
    refuse(EVIDENCE_INVALID, role=role, reason="unreadable_frame", count=1)


def _unique_rows(role: FeedRole, rows: Iterable[_Row]) -> dict[_Key, _Row]:
    columns = OVERLAP_RELATIONS[role].key
    indexed: dict[_Key, _Row] = {}
    duplicates: set[_Key] = set()
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in indexed:
            duplicates.add(key)
        else:
            indexed[key] = row
    _require(role, "duplicate_key", len(duplicates))
    return indexed


def _violations(
    role: FeedRole,
    rows: Iterable[_Row],
    *checks: tuple[str, Callable[[_Row], bool]],
) -> None:
    for reason, violation in checks:
        _require(role, reason, sum(violation(row) for row in rows))


def _decision_pair(
    unit: str,
    plan: OverlapPlan,
    decisions: dict[_Key, _Row],
) -> tuple[_Row | None, _Row | None]:
    return decisions.get((unit, plan.a.target)), decisions.get((unit, plan.b.target))


def _last_eligibility(row: _Row) -> datetime:
    return max(row["eligibility_at_a"], row["eligibility_at_b"])


def _invalid_disposition(row: _Row, factor: OverlapFactor, *, sdk: bool) -> bool:
    status, selected, arm, weight = (
        row["status"],
        row["selected_experiment"],
        row["arm"],
        row["assigned_weight"],
    )
    if (
        sdk
        and factor.layer is None
        and (status == "unallocated" or selected not in (None, factor.experiment))
    ):
        return True
    if status in ("held_out", "unallocated"):
        return selected is not None or arm is not None or weight is not None
    if status == "not_enrolled":
        return selected is None or arm is not None or weight is not None
    if selected is None or arm is None or weight is None or not 0 < weight <= MAX_EXACT_INTEGER:
        return True
    if selected != factor.experiment:
        return False
    return (arm, weight) not in ((factor.off, factor.off_weight), (factor.on, factor.on_weight))


def _disposition(decision: _Row, factor: OverlapFactor) -> GateDisposition:
    if decision["selected_experiment"] not in (None, factor.experiment):
        return "other_selected"
    return cast("GateDisposition", decision["status"])


def _validate_entry(
    plan: OverlapPlan,
    rosters: dict[_Key, _Row],
    decisions: dict[_Key, _Row],
    deliveries: dict[_Key, _Row],
) -> None:
    roster = rosters.values()
    _require("decisions", "orphan_unit", sum((unit,) not in rosters for unit, _ in decisions))
    _require(
        "deliveries",
        "orphan_unit",
        sum((row["unit_id"],) not in rosters for row in deliveries.values()),
    )
    # A negative decision is still an attempt. Require exactly one row in both directions.
    for reason, eligible in (("unexpected_decision", False), ("missing_decision", True)):
        for side, factor in zip(FACTOR_IDS, (plan.a, plan.b), strict=True):
            field = f"eligible_{side.lower()}"
            count = sum(
                row[field] is eligible
                and ((row["unit_id"], factor.target) in decisions) != eligible
                for row in roster
            )
            if count:
                refuse_assignment_invalid(reason, details=(side, count))
    _violations(
        "roster",
        roster,
        ("entry_outside_window", lambda r: not plan.entry_open <= r["entry_at"] < plan.start),
        (
            "eligibility_before_entry",
            lambda r: min(r["eligibility_at_a"], r["eligibility_at_b"]) < r["entry_at"],
        ),
        (
            "eligibility_after_start",
            lambda r: _last_eligibility(r) > plan.start,
        ),
        (
            "eligibility_not_before_either_decision",
            lambda r: any(
                _last_eligibility(r) >= d["decided_at"]
                for d in _decision_pair(r["unit_id"], plan, decisions)
                if d is not None
            ),
        ),
    )
    factors = {plan.a.target: plan.a, plan.b.target: plan.b}
    archives = dict(zip(factors, plan.law.archive_sha256, strict=True))
    _violations(
        "decisions",
        decisions.values(),
        ("revision_mismatch", lambda d: d["revision"] != factors[d["target"]].revision),
        ("archive_mismatch", lambda d: d["archive_sha256"] != archives[d["target"]]),
        ("decision_after_start", lambda d: d["decided_at"] > plan.start),
        ("invalid_status", lambda d: d["status"] not in DECISION_STATUSES),
        (
            "invalid_disposition",
            lambda d: _invalid_disposition(
                d, factors[d["target"]], sdk=plan.law.origin == "sdk_v1"
            ),
        ),
    )
    if plan.law.gate_basis == "preassigned_roster":
        both = [r for r in roster if r["eligible_a"] and r["eligible_b"]]
        _violations(
            "roster",
            both,
            ("missing_gate", lambda r: any(r[c.name] is None for c in PREASSIGNED_GATE_COLUMNS)),
            (
                "gate_out_of_order",
                lambda r: (
                    r["gate_at_a"] < r["eligibility_at_a"]
                    or r["gate_at_b"] < r["eligibility_at_b"]
                    or max(r["gate_at_a"], r["gate_at_b"])
                    >= min(
                        decisions[(r["unit_id"], f.target)]["decided_at"] for f in factors.values()
                    )
                ),
            ),
            (
                "gate_mismatch",
                lambda r: any(
                    r[f"gate_{side.lower()}"]
                    != (_disposition(decisions[(r["unit_id"], f.target)], f) == "assigned")
                    for side, f in zip(FACTOR_IDS, factors.values(), strict=True)
                ),
            ),
        )


def _spine(
    plan: OverlapPlan,
    roster: Iterable[_Row],
    decisions: dict[_Key, _Row],
) -> tuple[tuple[int, int, int, int], tuple[int, ...], dict[str, CellId]]:
    eligibility = [0, 0, 0, 0]
    gates: Counter[tuple[GateDisposition, GateDisposition]] = Counter()
    spine: dict[str, CellId] = {}
    for row in roster:
        mask = row["eligible_a"] + 2 * row["eligible_b"]
        eligibility[mask] += 1
        if mask != 3:
            continue
        unit = row["unit_id"]
        a, b = decisions[(unit, plan.a.target)], decisions[(unit, plan.b.target)]
        da, db = _disposition(a, plan.a), _disposition(b, plan.b)
        gates[da, db] += 1
        if da == db == "assigned":
            spine[unit] = CELL_IDS[(a["arm"] == plan.a.on) + 2 * (b["arm"] == plan.b.on)]
    if plan.law.admission == "both_guaranteed":
        _require("decisions", "admission_count_mismatch", eligibility[3] - len(spine))
    return (
        (eligibility[0], eligibility[1], eligibility[2], eligibility[3]),
        tuple(gates[key] for key in GATE_KEYS),
        spine,
    )


def _delivery_counts(
    plan: OverlapPlan,
    capture: OverlapCapture,
    roster: dict[_Key, _Row],
    deliveries: Iterable[_Row],
    decisions: dict[_Key, _Row],
    spine: dict[str, CellId],
    assigned: Counter[CellId],
) -> tuple[OverlapDeliveryCount, ...]:
    _violations(
        "deliveries",
        deliveries,
        ("delivery_before_start", lambda d: d["delivered_at"] < plan.start),
        ("delivery_after_cutoff", lambda d: d["delivered_at"] > capture.cutoff),
        (
            "eligibility_not_before_delivery",
            lambda d: _last_eligibility(roster[(d["unit_id"],)]) >= d["delivered_at"],
        ),
    )
    factors = {plan.a.experiment: ("A", plan.a), plan.b.experiment: ("B", plan.b)}
    matching: Counter[tuple[DeliveryCellId, str]] = Counter()
    inconsistent: Counter[tuple[DeliveryCellId, str]] = Counter()
    delivered: Counter[tuple[DeliveryCellId, str]] = Counter()
    seen: set[tuple[str, str]] = set()
    for event in deliveries:
        if event["delivered_at"] >= plan.end:
            continue
        unit = event["unit_id"]
        side, factor = factors[event["experiment"]]
        decision = decisions.get((unit, factor.target))
        matches = (
            decision is not None
            and _disposition(decision, factor) == "assigned"
            and event["target"] == factor.target
            and event["revision"] == factor.revision
            and event["delivered_arm"] == decision["arm"]
        )
        cell: DeliveryCellId = spine.get(unit, "outside_target")
        if cell == "outside_target":
            inconsistent[cell, side] += not matches
            continue
        if (unit, side) not in seen:
            seen.add((unit, side))
            delivered[cell, side] += 1
        (matching if matches else inconsistent)[cell, side] += 1
    return tuple(
        OverlapDeliveryCount(
            cell=cell,
            factor=side,
            n_without_delivery=0
            if cell == "outside_target"
            else assigned[cell] - delivered[cell, side],
            n_matching=matching[cell, side],
            n_inconsistent=inconsistent[cell, side],
        )
        for cell, side in DELIVERY_KEYS
    )


def _numeric_range(metric: str, quantity: str) -> NoReturn:
    refuse(NUMERIC_RANGE, metric=metric, quantity=quantity)


def _centered(values: list[float], metric: str) -> tuple[float, float, float]:
    anchor, maximum = min(values), max(values)
    if anchor == maximum:
        return (0.0 if anchor == 0.0 else anchor), 0.0, 0.0
    if not isfinite(maximum - anchor):
        _numeric_range(metric, "centered_range")
    # Floats are dyadic rationals. Accumulate on their common integer grid
    # and round once, preserving the residual even when anchor + residual rounds.
    ratios = [value.as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    base, denominator = anchor.as_integer_ratio()
    base *= scale // denominator
    deltas = [numerator * (scale // denominator) - base for numerator, denominator in ratios]
    n = len(values)
    total = sum(deltas)
    divisor = n * scale
    delta = total / divisor
    if not delta or not isfinite(anchor + delta):
        _numeric_range(metric, "mean_delta")
    try:
        m2 = sum((n * residual - total) ** 2 for residual in deltas) / (divisor**2)
    except OverflowError:
        pass
    else:
        if m2 > 0:
            return (0.0 if anchor == 0.0 else anchor), delta, m2
    _numeric_range(metric, "m2")


def _cells(
    plan: OverlapPlan,
    capture: OverlapCapture,
    outcomes: dict[_Key, _Row],
    spine: dict[str, CellId],
    assigned: Counter[CellId],
) -> tuple[OverlapCell, ...]:
    metrics = {metric.name: metric for metric in plan.metrics}
    _violations(
        "outcomes",
        outcomes.values(),
        (
            "window_mismatch",
            lambda r: (r["window_start"], r["window_end"]) != (plan.start, plan.end),
        ),
        (
            "definition_mismatch",
            lambda r: r["definition_sha256"] != metrics[r["metric"]].definition_sha256,
        ),
        (
            "finalized_after_cutoff",
            lambda r: r["finalized_at"] is not None and r["finalized_at"] > capture.cutoff,
        ),
        (
            "finalized_before_end",
            lambda r: r["finalized_at"] is not None and r["finalized_at"] < plan.end,
        ),
        (
            "invalid_value",
            lambda r: (
                r["value"] is not None
                and (
                    not isfinite(r["value"])
                    or (metrics[r["metric"]].kind == "binary_mean" and r["value"] not in (0.0, 1.0))
                )
            ),
        ),
    )
    complete_feed = capture.outcomes_cover(plan)
    cells: list[OverlapCell] = []
    for metric in plan.metrics:
        binary = metric.kind == "binary_mean"
        observed: Counter[CellId] = Counter()
        successes: Counter[CellId] = Counter()
        pending: Counter[CellId] = Counter()
        values: dict[CellId, list[float]] = {cell: [] for cell in CELL_IDS}
        if complete_feed:
            for unit, cell in spine.items():
                row = outcomes.get((unit, metric.name))
                if row is None:
                    continue
                if row["finalized_at"] is None:
                    pending[cell] += 1
                    continue
                value = row["value"]
                if value is None:
                    continue
                observed[cell] += 1
                if binary:
                    successes[cell] += value == 1.0
                else:
                    values[cell].append(value)
        for cell in CELL_IDS:
            n, count = assigned[cell], observed[cell]
            n_pending = pending[cell] if complete_feed else n
            anchor = delta = m2 = None
            if count and count == n:
                anchor, delta, m2 = (
                    overlap_binary_moments(count, successes[cell])
                    if binary
                    else _centered(values[cell], metric.name)
                )
            cells.append(
                OverlapCell(
                    metric=metric.name,
                    cell=cell,
                    n_assigned=n,
                    n_observed=count,
                    n_missing=n - count - n_pending,
                    n_pending=n_pending,
                    n_success=successes[cell] if binary else None,
                    anchor=anchor,
                    mean_delta=delta,
                    m2=m2,
                )
            )
    return tuple(cells)


def capture_overlap_frame(
    plan: OverlapPlan,
    *,
    capture: OverlapCapture,
    roster: IntoDataFrame,
    decisions: IntoDataFrame,
    deliveries: IntoDataFrame,
    outcomes: IntoDataFrame,
) -> OverlapEvidence:
    """Validate four local snapshot frames and retain only their bounded evidence.

    The producer supplies coherent materialized frames under their declared
    immutable generation. No source rows or identifier samples leave capture.
    """
    if not isinstance(capture, OverlapCapture):
        refuse(CAPTURE_INVALID, role="snapshot", reason="invalid_value")
    capture.validate_for(plan)
    sources: dict[FeedRole, IntoDataFrame] = {
        "roster": roster,
        "decisions": decisions,
        "deliveries": deliveries,
        "outcomes": outcomes,
    }
    physical = {role: relation.columns for role, relation in OVERLAP_RELATIONS.items()}
    if plan.law.gate_basis == "preassigned_roster":
        physical["roster"] += PREASSIGNED_GATE_COLUMNS
    frames = {role: _frame(source, role, physical[role]) for role, source in sources.items()}
    rows = {
        role: _rows(frame, role, physical[role], nanos, plan)
        for role, (frame, nanos) in frames.items()
    }
    indexed = {role: _unique_rows(role, records) for role, records in rows.items()}
    entry, attempts, events = indexed["roster"], indexed["decisions"], indexed["deliveries"]
    _validate_entry(plan, entry, attempts, events)
    eligibility, gates, spine = _spine(plan, entry.values(), attempts)
    assigned = Counter(spine.values())
    delivery_counts = _delivery_counts(
        plan, capture, entry, events.values(), attempts, spine, assigned
    )
    cells = _cells(plan, capture, indexed["outcomes"], spine, assigned)
    return OverlapEvidence(
        plan=plan,
        capture=capture,
        eligibility_counts=eligibility,
        gate_counts=gates,
        delivery_counts=delivery_counts,
        cells=cells,
        ingress="frame",
    )
