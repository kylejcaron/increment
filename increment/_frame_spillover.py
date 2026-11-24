"""Reconcile the two stages of a spillover trial from eager dataframes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from math import fsum
from numbers import Real
from typing import TYPE_CHECKING, Any

import narwhals as nw
import numpy as np

from increment._spillover_design import (
    GroupProfile,
    Identifier,
    PolicyCell,
    SpilloverDesign,
    SpilloverPopulation,
    canonical_id,
    is_label,
    refuse_input,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._metric_specs import MetricSpec
    from increment._spillover_design import Spillover


def _as_frame(frame: IntoDataFrame | None, role: str) -> nw.DataFrame[Any]:
    prelaunch = role in ("roster", "offer_counts")
    code = "spillover.design.invalid" if prelaunch else "spillover.log.keys"
    if frame is None:
        refuse_input(code, role)
    try:
        df = nw.from_native(frame, eager_only=True)
    except (TypeError, ValueError):
        df = None
    if not isinstance(df, nw.DataFrame):
        refuse_input(code, role)
    return df


def _rows(frame: IntoDataFrame | None, role: str, *columns: str) -> Iterator[tuple[Any, ...]]:
    df = _as_frame(frame, role)
    code = (
        "spillover.design.invalid" if role in ("roster", "offer_counts") else "spillover.log.keys"
    )
    names = Counter(df.columns)
    if any(not is_label(c) or names[c] != 1 for c in columns) or len(set(columns)) != len(columns):
        refuse_input(code, role)
    if role == "outcomes":
        outcome_col = nw.col(columns[1])
        null_mask = outcome_col.is_null()
        if df.schema[columns[1]].is_float():
            null_mask = null_mask | outcome_col.is_nan()
        return (
            (person, None if missing else value)
            for person, value, missing in df.select(
                nw.col(columns[0]).alias("person"),
                outcome_col.alias("value"),
                null_mask.alias("missing"),
            ).iter_rows()
        )
    return df.select(*columns).iter_rows()


@dataclass(slots=True)
class _Roster:
    membership: dict[Identifier, Identifier]
    strata: dict[Identifier, str]


def _roster(
    frame: IntoDataFrame, unit: str, cluster: str, stratum: str | None, design: SpilloverDesign
) -> _Roster:
    membership: dict[Identifier, Identifier] = {}
    strata: dict[Identifier, str] = {}
    duplicate = False
    if stratum is None:
        if len(design.groups_per_policy) != 1:
            refuse_input("spillover.design.invalid", "stratum")
        sole_stratum = next(iter(design.groups_per_policy))
        rows = ((p, g, sole_stratum) for p, g in _rows(frame, "roster", unit, cluster))
    else:
        rows = _rows(frame, "roster", unit, cluster, stratum)
    for raw_person, raw_group, group_stratum in rows:
        person, group = canonical_id(raw_person), canonical_id(raw_group)
        if person is None or group is None or not is_label(group_stratum):
            refuse_input("spillover.design.invalid", "roster")
        previous_stratum = strata.setdefault(group, group_stratum)
        if person in membership or previous_stratum != group_stratum:
            duplicate = True
        membership[person] = group
    if (
        not membership
        or len({type(p) for p in membership}) != 1
        or len({type(g) for g in strata}) != 1
    ):
        refuse_input("spillover.design.invalid", "roster")
    if duplicate:
        refuse_input("spillover.design.duplicate", "roster")
    return _Roster(membership, strata)


def _profiles(
    frame: IntoDataFrame, cluster: str, roster: _Roster, design: SpilloverDesign
) -> tuple[GroupProfile, ...]:
    counts: dict[tuple[Identifier, str], int] = {}
    duplicate = bad_law = False
    group_type = type(next(iter(roster.strata)))
    for raw_group, policy, offers in _rows(frame, "offer_counts", cluster, "policy", "offers"):
        group = canonical_id(raw_group)
        if group is None or type(group) is not group_type or not is_label(policy):
            refuse_input("spillover.design.invalid", "offer_counts")
        key = (group, policy)
        if key in counts:
            duplicate = True
        if group not in roster.strata or policy not in design.policies:
            bad_law = True
        counts[key] = offers
    if duplicate:
        refuse_input("spillover.design.duplicate", "offer_counts")
    if bad_law or len(counts) != len(roster.strata) * len(design.policies):
        refuse_input("spillover.design.quota", "offer_counts")
    sizes = Counter(roster.membership.values())
    return tuple(
        GroupProfile(g, roster.strata[g], sizes[g], tuple(counts[g, p] for p in design.policies))
        for g in sorted(roster.strata, key=lambda k: (roster.strata[k], k))
    )


def _digest(roster: _Roster, policies: tuple[str, ...], profiles: tuple[GroupProfile, ...]) -> str:
    # Newline-delimited JSON arrays preserve key domains, delimiters and Unicode exactly.
    digest = hashlib.sha256(b"spillover-frame-v1\n")

    def record(row: tuple[object, ...]) -> None:
        digest.update(json.dumps(row, ensure_ascii=True, separators=(",", ":")).encode("ascii"))
        digest.update(b"\n")

    record(policies)
    for person in sorted(roster.membership):
        record((person, roster.membership[person]))
    for g in profiles:
        record((g.group_id, g.stratum, g.size, g.offers))
    return digest.hexdigest()


def _keyed_log(
    frame: IntoDataFrame | None,
    role: str,
    expected: dict[Identifier, Any],
    key_col: str,
    value_col: str,
) -> dict[Identifier, Any]:
    records: dict[Identifier, Any] = {}
    for key, value in _rows(frame, role, key_col, value_col):
        key = canonical_id(key)
        if key is None or key not in expected or key in records:
            refuse_input("spillover.log.keys", role)
        records[key] = value
    if len(records) != len(expected):
        refuse_input("spillover.log.keys", role)
    return records


def _group_policies(
    frame: IntoDataFrame | None, cluster: str, roster: _Roster, design: SpilloverDesign
) -> dict[Identifier, str]:
    policies = _keyed_log(frame, "group_policy", roster.strata, cluster, "policy")
    allocation: Counter[tuple[str, str]] = Counter()
    for group, policy in policies.items():
        if not is_label(policy) or policy not in design.policies:
            refuse_input("spillover.log.keys", "group_policy")
        allocation[roster.strata[group], policy] += 1
    if any(
        allocation[s, p] != n
        for s, quotas in design.groups_per_policy.items()
        for p, n in quotas.items()
    ):
        refuse_input("spillover.log.quota", "group_policy")
    return policies


def _binary_outcome(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if (
        not isinstance(value, (Real, Decimal))
        or isinstance(value, np.timedelta64)
        or value not in (0, 1)
    ):
        refuse_input("spillover.log.binary", "outcomes")
    return bool(value)


def _verify_offers(
    assignments: dict[Identifier, Any],
    membership: dict[Identifier, Identifier],
    population: SpilloverPopulation,
    policy: dict[Identifier, str],
    design: Spillover,
) -> Counter[Identifier]:
    by_group: Counter[Identifier] = Counter()
    for person, label in assignments.items():
        if not is_label(label) or label not in (design.control_group, design.offered_group):
            refuse_input("spillover.log.keys", "offered")
        if label == design.offered_group:
            by_group[membership[person]] += 1
    index = {p: i for i, p in enumerate(population.design.policies)}
    for g in population.groups:
        if by_group[g.group_id] != g.offers[index[policy[g.group_id]]]:
            refuse_input("spillover.log.quota", "offered")
    return by_group


def _untreated_events(
    outcomes: dict[Identifier, Any],
    membership: dict[Identifier, Identifier],
    assignments: dict[Identifier, str],
    control_group: str,
) -> Counter[Identifier]:
    if any(value is None and assignments[p] == control_group for p, value in outcomes.items()):
        refuse_input("spillover.log.keys", "outcomes")
    events: Counter[Identifier] = Counter()
    for person, value in outcomes.items():
        if assignments[person] == control_group and _binary_outcome(value):
            events[membership[person]] += 1
    return events


@dataclass(slots=True)
class _Cell:
    totals: list[float] = field(default_factory=list)
    unoffered: int = 0
    events: int = 0

    def finish(self, stratum: str, policy: str) -> PolicyCell:
        anchor = self.totals[0]
        center = fsum(t - anchor for t in self.totals) / len(self.totals)
        ss = fsum(((t - anchor) - center) ** 2 for t in self.totals)
        return PolicyCell(
            stratum, policy, len(self.totals), anchor + center, ss, self.unoffered, self.events
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class FrameSpilloverSource:
    """One row per eligible person, plus a prospective schedule and realized group policies."""

    frame: nw.DataFrame[Any]
    unit: str
    group: str
    cluster: str
    stratum: str | None
    design: Spillover
    outcome: str
    offer_counts: IntoDataFrame
    group_policy: IntoDataFrame | None
    recipe: tuple[tuple[str, str], ...]

    def _capture(self, design: SpilloverDesign) -> tuple[_Roster, SpilloverPopulation]:
        if not isinstance(design, SpilloverDesign):
            refuse_input("spillover.design.required", "design")
        roster = _roster(self.frame, self.unit, self.cluster, self.stratum, design)
        profiles = _profiles(self.offer_counts, self.cluster, roster, design)
        identity = _digest(roster, design.policies, profiles)
        return roster, SpilloverPopulation(design, profiles, identity)

    def freeze(self, law: SpilloverDesign) -> SpilloverPopulation:
        """Capture eligible group profiles without accessing either assignment or outcomes."""
        return self._capture(law)[1]

    def observe(self, population: SpilloverPopulation) -> tuple[PolicyCell, ...]:
        """Validate both assignment stages and reduce the untreated outcomes by policy and stratum."""
        roster, current = self._capture(population.design)
        if current != population:
            refuse_input("spillover.roster.changed", "roster")
        policies = _group_policies(self.group_policy, self.cluster, roster, population.design)
        assignments = _keyed_log(self.frame, "offered", roster.membership, self.unit, self.group)
        n_offered = _verify_offers(
            assignments, roster.membership, population, policies, self.design
        )
        outcomes = _keyed_log(self.frame, "outcomes", roster.membership, self.unit, self.outcome)
        events = _untreated_events(
            outcomes, roster.membership, assignments, self.design.control_group
        )
        compared = population.design.compared_policies
        cells = {(s, p): _Cell() for s in population.design.groups_per_policy for p in compared}
        for g in population.groups:
            p = policies[g.group_id]
            if p in compared:
                unoffered = g.size - n_offered[g.group_id]
                count = events[g.group_id]
                cell = cells[g.stratum, p]
                cell.totals.append((g.size * count) / unoffered)
                cell.unoffered += unoffered
                cell.events += count
        return tuple(cells[k].finish(*k) for k in sorted(cells))


def summary_spillover_source(
    frame: IntoDataFrame,
    *,
    unit: str,
    group: str,
    cluster: str,
    stratum: str | None,
    design: Spillover,
    metric: MetricSpec,
    offer_counts: IntoDataFrame,
    group_policy: IntoDataFrame | None,
) -> FrameSpilloverSource:
    """Bind a precomputed conversion to the two recorded assignment stages."""
    outcome = metric.value_column or metric.name
    recipe: tuple[tuple[str, str], ...] = (
        ("source", "frame"),
        ("shape", "summary"),
        ("unit", unit),
        ("group", group),
        ("cluster", cluster),
        ("control_group", design.control_group),
        ("offered_group", design.offered_group),
        ("metric", metric.name),
        ("metric_type", metric.type),
        ("window_days", str(metric.window_days)),
        ("value_column", outcome),
    )
    if stratum is not None:
        recipe += (("stratum", stratum),)
    return FrameSpilloverSource(
        _as_frame(frame, "roster"),
        unit,
        group,
        cluster,
        stratum,
        design,
        outcome,
        offer_counts,
        group_policy,
        recipe,
    )
