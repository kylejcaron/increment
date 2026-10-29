"""Backend-neutral assignment ledger validation and reduction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from increment._assignment_refusals import refuse_assignment_invalid
from increment._identity import canonical_id_strings


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


_REQUIRED = frozenset({"experiment", "arm", "version"})
_OPTIONAL = frozenset({"cluster", "period", "layer", "allocated_at", "exposed_at", "triggered_at"})
_TIMES = ("allocated_at", "exposed_at", "triggered_at")


@dataclass(frozen=True, slots=True)
class AssignmentRow:
    unit: str
    experiment: str
    arm: str
    period: str | None
    version: int
    layer: str | None
    allocated_at: datetime | date | None
    exposed_at: datetime | date | None
    triggered_at: datetime | date | None
    cluster: str | None = None


@dataclass(frozen=True, slots=True)
class ReducedAssignment:
    experiment: str
    population: str
    versions: tuple[int, ...]
    periods: tuple[str | None, ...]
    arm_counts: tuple[tuple[int, str | None, str, int], ...]
    rows: tuple[AssignmentRow, ...]
    independence_grain: str
    replay_status: str = "not_certified"


@dataclass(frozen=True, slots=True)
class _Selection:
    experiment: str
    version: int | None
    period: str | None
    key: str | None


@dataclass(slots=True)
class _ValidationState:
    unique: set[tuple[object, ...]]
    unit_cluster: dict[str, str | None]
    cluster_arms: dict[tuple[object, ...], str]


def _project_record(
    record: Mapping[str, Any],
    roles: Mapping[str, str],
    selection: _Selection,
    state: _ValidationState,
) -> AssignmentRow | None:
    if record.get(roles["experiment"]) != selection.experiment:
        return None
    row_version = record.get(roles["version"])
    if row_version is None:
        refuse_assignment_invalid("missing_version")
    raw_period = record.get(roles["period"]) if "period" in roles else None
    period = None if _is_missing(raw_period) else raw_period
    if selection.version is not None and row_version != selection.version:
        return None
    if selection.period is not None and period != selection.period:
        return None
    unit = record.get(roles["unit"]) if "unit" in roles else None
    cluster = record.get(roles["cluster"]) if "cluster" in roles else None
    identity = cluster if selection.key == "cluster" else unit
    if identity is None:
        refuse_assignment_invalid("missing_identity")
    arm = record.get(roles["arm"])
    if not isinstance(arm, str) or not arm:
        refuse_assignment_invalid("invalid_arm", details=arm)
    key = (unit if unit is not None else identity, selection.experiment, row_version, period)
    if key in state.unique:
        refuse_assignment_invalid("duplicate_assignment_key", details=key)
    state.unique.add(key)
    if cluster is not None:
        if unit is not None:
            previous = state.unit_cluster.setdefault(str(unit), str(cluster))
            if previous != str(cluster):
                refuse_assignment_invalid("inconsistent_unit_cluster_membership")
        cluster_key = (str(cluster), selection.experiment, row_version, period)
        previous_arm = state.cluster_arms.setdefault(cluster_key, arm)
        if previous_arm != arm:
            refuse_assignment_invalid("conflicting_cluster_arms", details=cluster_key)
    times: dict[str, datetime | date | None] = {}
    for name in _TIMES:
        raw_value = record.get(roles[name]) if name in roles else None
        value = None if _is_missing(raw_value) else raw_value
        if value is not None and not isinstance(value, (datetime, date)):
            refuse_assignment_invalid("invalid_event_timestamp", details=name)
        times[name] = value
    layer = record.get(roles["layer"]) if "layer" in roles else None
    return AssignmentRow(
        unit=str(unit) if unit is not None else "",
        experiment=selection.experiment,
        arm=arm,
        period=str(period) if period is not None else None,
        version=int(row_version),
        layer=str(layer) if layer is not None else None,
        allocated_at=times["allocated_at"],
        exposed_at=times["exposed_at"],
        triggered_at=times["triggered_at"],
        cluster=str(cluster) if cluster is not None else None,
    )


def validate_roles(roles: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(roles, Mapping):
        refuse_assignment_invalid("invalid_roles", details=roles)
    mapped = dict(roles)
    missing = _REQUIRED - mapped.keys()
    unknown = mapped.keys() - _REQUIRED - _OPTIONAL - {"unit"}
    if missing or unknown:
        refuse_assignment_invalid(
            "invalid_roles", details={"missing": sorted(missing), "unknown": sorted(unknown)}
        )
    if "unit" not in mapped and "cluster" not in mapped:
        refuse_assignment_invalid("missing_assignment_identity_role")
    if len(set(mapped.values())) != len(mapped):
        refuse_assignment_invalid("duplicate_assignment_role_column")
    return mapped


def reduce_rows(
    records: Any,
    roles: Mapping[str, str],
    *,
    experiment: str,
    population: str,
    version: int | None = None,
    period: str | None = None,
    all_periods: bool = False,
    key: str | None = None,
) -> ReducedAssignment:
    mapped = validate_roles(roles)
    if key not in (None, "unit", "cluster"):
        refuse_assignment_invalid("invalid_key", details=key)
    if key == "cluster" and "cluster" not in mapped:
        refuse_assignment_invalid("missing_cluster_role")
    if key != "cluster" and "unit" not in mapped:
        refuse_assignment_invalid("missing_unit_role")
    if population not in {"assigned", "triggered"}:
        refuse_assignment_invalid("invalid_population", details=population)
    if all_periods and period is not None:
        refuse_assignment_invalid("period_and_all_periods_conflict")
    selection = _Selection(experiment, version, period, key)
    selected_records = []
    for record in records:
        if record.get(mapped["experiment"]) != experiment:
            continue
        row_version = record.get(mapped["version"])
        if row_version is None:
            refuse_assignment_invalid("missing_version")
        if version is not None and row_version != version:
            continue
        raw_period = record.get(mapped["period"]) if "period" in mapped else None
        row_period = None if _is_missing(raw_period) else raw_period
        if period is not None and row_period != period:
            continue
        selected_records.append(dict(record))
    for identity_role in ("unit", "cluster"):
        column = mapped.get(identity_role)
        if column is None:
            continue
        positions = [
            index for index, row in enumerate(selected_records) if row.get(column) is not None
        ]
        native = np.asarray([selected_records[index][column] for index in positions], dtype=object)
        canonical = canonical_id_strings(native, what=f"assignment {identity_role} identities")
        for index, value in zip(positions, canonical, strict=True):
            selected_records[index][column] = value
    state = _ValidationState(set(), {}, {})
    projected = [
        row
        for record in selected_records
        if (row := _project_record(record, mapped, selection, state)) is not None
    ]
    if not projected:
        refuse_assignment_invalid("empty_assignment_selection")
    versions = tuple(sorted({row.version for row in projected}))
    periods = tuple(
        sorted(
            {row.period for row in projected}, key=lambda value: (value is not None, value or "")
        )
    )
    if version is None and len(versions) > 1:
        refuse_assignment_invalid("mixed_allocation_versions", details=versions)
    if not all_periods and period is None and len(periods) > 1:
        refuse_assignment_invalid("mixed_assignment_periods", details=periods)
    if population == "triggered":
        projected = [row for row in projected if row.triggered_at is not None]
    if key == "cluster":
        counts = Counter((row.version, row.period, row.arm, row.cluster) for row in projected)
        counts = Counter((version, period, arm) for version, period, arm, _cluster in counts)
    else:
        counts = Counter((row.version, row.period, row.arm) for row in projected)

    def order(row: AssignmentRow) -> tuple[int, str, str, str]:
        return (row.version, row.period or "", row.unit, row.cluster or "")

    projected.sort(key=order)
    return ReducedAssignment(
        experiment=experiment,
        population=population,
        versions=versions,
        periods=periods,
        arm_counts=tuple(
            sorted(
                ((v, p, arm, n) for (v, p, arm), n in counts.items()),
                key=lambda item: (item[0], item[1] is not None, item[1] or "", item[2]),
            )
        ),
        rows=tuple(projected),
        independence_grain="cluster" if key == "cluster" else "unit",
    )


def exposure_rows(
    reduced: ReducedAssignment,
    *,
    period_order: tuple[str, ...],
    key: str | None,
) -> tuple[tuple[str, str, str, str | None], ...]:
    """Join each selected assignment period to the next period's arm."""
    if not period_order or len(set(period_order)) != len(period_order):
        refuse_assignment_invalid("invalid_period_order")
    positions = {period: index for index, period in enumerate(period_order)}
    grouped: dict[tuple[str, int], dict[str, str]] = {}
    for row in reduced.rows:
        if row.period not in positions:
            refuse_assignment_invalid("period_order_missing_assignment", details=row.period)
        identity = row.cluster if key == "cluster" else row.unit
        if not identity:
            refuse_assignment_invalid("missing_identity")
        grouped.setdefault((identity, row.version), {})[row.period] = row.arm
    output = []
    for (identity, _version), assignments in sorted(grouped.items()):
        for period in period_order:
            arm = assignments.get(period)
            if arm is None:
                continue
            following = (
                period_order[positions[period] + 1]
                if positions[period] + 1 < len(period_order)
                else None
            )
            output.append(
                (identity, period, arm, assignments.get(following) if following else None)
            )
    return tuple(output)
