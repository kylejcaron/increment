"""Ibis assignment ledger adapter, isolated to the query layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from increment._assignment_ledger_contract import (
    ReducedAssignment,
    _is_missing,
    validate_roles,
)
from increment._evidence_refusals import refuse_assignment_invalid
from increment._frame_ledger import FrameLedger


def _validate_selected(
    relation: Any, roles: Mapping[str, str], key: str | None
) -> tuple[str, str, str | None, str]:
    identity = roles["cluster" if key == "cluster" else "unit"]
    version_column = roles["version"]
    period_column = roles.get("period")
    arm_column = roles["arm"]
    if not relation.filter(relation[identity].isnull()).limit(1).execute(limit=None).empty:
        refuse_assignment_invalid("missing_identity")
    if not relation.filter(relation[version_column].isnull()).limit(1).execute(limit=None).empty:
        refuse_assignment_invalid("missing_version")
    if not relation.filter(relation[arm_column].isnull()).limit(1).execute(limit=None).empty:
        refuse_assignment_invalid("invalid_arm")
    arm_type = relation[arm_column].type()
    if (
        not arm_type.is_string()
        or not relation.filter(relation[arm_column] == "").limit(1).execute(limit=None).empty
    ):
        refuse_assignment_invalid("invalid_arm")
    for name in ("allocated_at", "exposed_at", "triggered_at"):
        column = roles.get(name)
        if column is None:
            continue
        dtype = relation[column].type()
        if not dtype.is_null() and not (dtype.is_date() or dtype.is_timestamp()):
            refuse_assignment_invalid("invalid_event_timestamp", details=name)
    if "unit" in roles and "cluster" in roles:
        memberships = relation.filter(relation[roles["cluster"]].notnull())
        membership = memberships.group_by(roles["unit"]).aggregate(
            n_clusters=memberships[roles["cluster"]].nunique()
        )
        if not membership.filter(membership.n_clusters > 1).limit(1).execute(limit=None).empty:
            refuse_assignment_invalid("inconsistent_unit_cluster_membership")
    duplicate_key = roles.get("unit", identity)
    duplicate_keys = [duplicate_key, version_column]
    if period_column is not None:
        duplicate_keys.append(period_column)
    duplicates = relation.group_by(duplicate_keys).aggregate(n=relation.count())
    if not duplicates.filter(duplicates.n > 1).limit(1).execute(limit=None).empty:
        refuse_assignment_invalid("duplicate_assignment_key")
    if key == "cluster":
        cluster_keys = [identity, version_column]
        if period_column is not None:
            cluster_keys.append(period_column)
        mixed = relation.group_by(cluster_keys).aggregate(n_arms=relation[arm_column].nunique())
        if not mixed.filter(mixed.n_arms > 1).limit(1).execute(limit=None).empty:
            refuse_assignment_invalid("conflicting_cluster_arms")
    return identity, version_column, period_column, arm_column


class Ledger:
    """Validate and reduce an identity-bearing warehouse assignment relation."""

    __slots__ = ("_relation", "roles")

    def __init__(self, relation: Any, roles: Mapping[str, str]) -> None:
        self.roles = validate_roles(roles)
        self._relation = relation
        schema = relation.schema()
        columns = set(schema.names)
        missing = set(self.roles.values()) - columns
        if missing:
            refuse_assignment_invalid("relation_columns_missing", details=sorted(missing))

    def _selected(self, experiment: str, version: int | None, period: str | None) -> Any:
        relation = self._relation.filter(self._relation[self.roles["experiment"]] == experiment)
        if version is not None:
            relation = relation.filter(relation[self.roles["version"]] == version)
        if period is not None:
            if "period" in self.roles:
                relation = relation.filter(relation[self.roles["period"]] == period)
            else:
                relation = relation.limit(0)
        return relation

    def reduce(
        self,
        *,
        experiment: str,
        population: str,
        version: int | None = None,
        period: str | None = None,
        all_periods: bool = False,
        key: str | None = None,
    ) -> ReducedAssignment:
        """Filter and aggregate assignment counts in the warehouse."""
        if key not in (None, "unit", "cluster"):
            refuse_assignment_invalid("invalid_key", details=key)
        if key == "cluster" and "cluster" not in self.roles:
            refuse_assignment_invalid("missing_cluster_role")
        if key != "cluster" and "unit" not in self.roles:
            refuse_assignment_invalid("missing_unit_role")
        if population not in {"assigned", "triggered"}:
            refuse_assignment_invalid("invalid_population", details=population)
        if all_periods and period is not None:
            refuse_assignment_invalid("period_and_all_periods_conflict")
        selected = self._selected(experiment, version, period)
        identity, version_column, period_column, arm_column = _validate_selected(
            selected, self.roles, key
        )
        metadata_keys = [version_column]
        if period_column is not None:
            metadata_keys.append(period_column)
        metadata = selected.group_by(metadata_keys).aggregate(n=selected.count())
        metadata_records = metadata.execute(limit=None).to_dict(orient="records")
        if not metadata_records:
            refuse_assignment_invalid("empty_assignment_selection")
        versions = tuple(sorted({int(row[version_column]) for row in metadata_records}))
        periods = tuple(
            sorted(
                {
                    None
                    if period_column is None or _is_missing(row.get(period_column))
                    else str(row[period_column])
                    for row in metadata_records
                },
                key=lambda value: (value is not None, value or ""),
            )
        )
        if version is None and len(versions) > 1:
            refuse_assignment_invalid("mixed_allocation_versions", details=versions)
        if not all_periods and period is None and len(periods) > 1:
            refuse_assignment_invalid("mixed_assignment_periods", details=periods)
        count_relation = selected
        if population == "triggered":
            count_relation = (
                selected.filter(selected[self.roles["triggered_at"]].notnull())
                if "triggered_at" in self.roles
                else selected.limit(0)
            )
        count_keys = [version_column]
        if period_column is not None:
            count_keys.append(period_column)
        count_keys.append(arm_column)
        identity_count = (
            count_relation[identity].nunique() if key == "cluster" else count_relation.count()
        )
        grouped = count_relation.group_by(count_keys).aggregate(assignment_count=identity_count)
        records = grouped.execute(limit=None).to_dict(orient="records")
        counts = tuple(
            sorted(
                [
                    (
                        int(row[version_column]),
                        None
                        if period_column is None or _is_missing(row.get(period_column))
                        else str(row[period_column]),
                        str(row[arm_column]),
                        int(row["assignment_count"]),
                    )
                    for row in records
                ],
                key=lambda item: (
                    item[0],
                    item[1] is not None,
                    item[1] or "",
                    item[2],
                ),
            )
        )
        return ReducedAssignment(
            experiment=experiment,
            population=population,
            versions=versions,
            periods=periods,
            arm_counts=counts,
            rows=(),
            independence_grain="cluster" if key == "cluster" else "unit",
        )

    def rows(
        self,
        *,
        experiment: str,
        population: str,
        version: int | None = None,
        period: str | None = None,
        all_periods: bool = False,
        key: str | None = None,
    ) -> ReducedAssignment:
        """Materialize full assignment rows only for callers that explicitly request them."""
        selected = self._selected(experiment, version, period)
        return FrameLedger(selected.execute(limit=None), self.roles).reduce(
            experiment=experiment,
            population=population,
            version=version,
            period=period,
            all_periods=all_periods,
            key=key,
        )

    def exposure(
        self,
        *,
        experiment: str,
        version: int,
        period_order: tuple[str, ...],
        key: str | None = None,
    ) -> tuple[tuple[str, str, str, str | None], ...]:
        """Join explicit full-row materialization and return bounded exposure rows."""
        from increment._assignment_ledger_contract import exposure_rows

        reduced = self.rows(
            experiment=experiment,
            population="assigned",
            version=version,
            all_periods=True,
            key=key,
        )
        return exposure_rows(reduced, period_order=period_order, key=key)
