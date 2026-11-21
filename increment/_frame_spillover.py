"""Reconcile the two stages of a spillover trial from eager dataframes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from math import fsum
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
    integer_scalar,
    is_label,
    refuse_input,
)

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

_COLUMNS = {
    "roster": ("person_id", "group_id", "stratum"),
    "offer_counts": ("group_id", "policy", "offers"),
    "group_policy": ("group_id", "policy"),
    "offered": ("person_id", "offered"),
    "outcomes": ("person_id", "outcome"),
}


def _rows(frame: IntoDataFrame | None, role: str) -> Iterator[tuple[Any, ...]]:
    prelaunch = role in ("roster", "offer_counts")
    code = "spillover.design.invalid" if prelaunch else "spillover.log.keys"
    if frame is None:
        refuse_input(code, role)
    try:
        df = nw.from_native(frame, eager_only=True)
    except (TypeError, ValueError):
        df = None
    columns = _COLUMNS[role]
    if not isinstance(df, nw.DataFrame):
        refuse_input(code, role)
    names = Counter(df.columns)
    if any(names[c] != 1 for c in columns):
        refuse_input(code, role)
    if role == "outcomes":
        # Preserve the backend's null mask (including pandas nullable integer/boolean arrays).
        return (
            (person, None if missing else value)
            for person, value, missing in df.select(
                *columns, nw.col("outcome").is_null().alias("_missing")
            ).iter_rows()
        )
    return df.select(*columns).iter_rows()


@dataclass(slots=True)
class _Roster:
    membership: dict[Identifier, Identifier]
    strata: dict[Identifier, str]


def _roster(frame: IntoDataFrame) -> _Roster:
    membership: dict[Identifier, Identifier] = {}
    strata: dict[Identifier, str] = {}
    duplicate = False
    for raw_person, raw_group, stratum in _rows(frame, "roster"):
        person, group = canonical_id(raw_person), canonical_id(raw_group)
        if person is None or group is None or not is_label(stratum):
            refuse_input("spillover.design.invalid", "roster")
        previous_stratum = strata.setdefault(group, stratum)
        if person in membership or previous_stratum != stratum:
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
    frame: IntoDataFrame, roster: _Roster, design: SpilloverDesign
) -> tuple[GroupProfile, ...]:
    counts: dict[tuple[Identifier, str], int] = {}
    duplicate = bad_law = False
    group_type = type(next(iter(roster.strata)))
    for raw_group, policy, offers in _rows(frame, "offer_counts"):
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
    frame: IntoDataFrame | None, role: str, expected: dict[Identifier, Any]
) -> dict[Identifier, Any]:
    records: dict[Identifier, Any] = {}
    for key, value in _rows(frame, role):
        key = canonical_id(key)
        if key is None or key not in expected or key in records:
            refuse_input("spillover.log.keys", role)
        records[key] = value
    if role != "outcomes" and len(records) != len(expected):
        refuse_input("spillover.log.keys", role)
    return records


def _group_policies(
    frame: IntoDataFrame | None, roster: _Roster, design: SpilloverDesign
) -> dict[Identifier, str]:
    policies = _keyed_log(frame, "group_policy", roster.strata)
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


def _binary(value: object, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if not integer_scalar(value) or value not in (0, 1):
        refuse_input("spillover.log.binary", role)
    return bool(value)


def _verify_offers(
    frame: IntoDataFrame | None,
    membership: dict[Identifier, Identifier],
    population: SpilloverPopulation,
    policy: dict[Identifier, str],
) -> tuple[dict[Identifier, bool], Counter[Identifier]]:
    offered = _keyed_log(frame, "offered", membership)
    by_group: Counter[Identifier] = Counter()
    for person, value in offered.items():
        offered[person] = _binary(value, "offered")
        if offered[person]:
            by_group[membership[person]] += 1
    index = {p: i for i, p in enumerate(population.design.policies)}
    for g in population.groups:
        if by_group[g.group_id] != g.offers[index[policy[g.group_id]]]:
            refuse_input("spillover.log.quota", "offered")
    return offered, by_group


def _untreated_events(
    frame: IntoDataFrame | None,
    membership: dict[Identifier, Identifier],
    offered: dict[Identifier, bool],
) -> Counter[Identifier]:
    outcomes = _keyed_log(frame, "outcomes", membership)
    if any(not flag and outcomes.get(person) is None for person, flag in offered.items()):
        refuse_input("spillover.log.keys", "outcomes")
    events: Counter[Identifier] = Counter()
    for person, value in outcomes.items():
        if value is not None and _binary(value, "outcomes") and not offered[person]:
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
    """Five input roles; planning uses only the eligible roster and prospective offer counts."""

    roster: IntoDataFrame
    offer_counts: IntoDataFrame
    group_policy: IntoDataFrame | None = None
    offered: IntoDataFrame | None = None
    outcomes: IntoDataFrame | None = None

    def _capture(self, design: SpilloverDesign) -> tuple[_Roster, SpilloverPopulation]:
        if not isinstance(design, SpilloverDesign):
            refuse_input("spillover.design.required", "design")
        roster = _roster(self.roster)
        profiles = _profiles(self.offer_counts, roster, design)
        identity = _digest(roster, design.policies, profiles)
        return roster, SpilloverPopulation(design, profiles, identity)

    def freeze(self, design: SpilloverDesign) -> SpilloverPopulation:
        """Capture eligible group profiles without accessing either assignment or outcomes."""
        return self._capture(design)[1]

    def observe(self, population: SpilloverPopulation) -> tuple[PolicyCell, ...]:
        """Validate both assignment stages and reduce the untreated outcomes by policy and stratum."""
        roster, current = self._capture(population.design)
        if current != population:
            refuse_input("spillover.roster.changed", "roster")
        policies = _group_policies(self.group_policy, roster, population.design)
        offered, n_offered = _verify_offers(self.offered, roster.membership, population, policies)
        events = _untreated_events(self.outcomes, roster.membership, offered)
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
