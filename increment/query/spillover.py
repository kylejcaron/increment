"""Capture the two randomizations and reduce untreated outcomes inside the warehouse."""

# Warehouse availability must be checked before importing optional dependencies.
# ruff: noqa: E402

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from increment._optional import require_warehouse

require_warehouse("increment.query.spillover")

from ibis import Table
from ibis.backends.sql import SQLBackend
from ibis.common.exceptions import TableNotFound

from increment._compile_gaps import translating_compile_gaps
from increment._spillover_design import (
    GroupProfile,
    PolicyCell,
    SpilloverDesign,
    SpilloverPopulation,
    refuse_input,
)
from increment.query.artifact_ports import SQL_DIGEST_FORMAT
from increment.query.session import (
    WarehouseSession,
    _require_warehouse_digest_verifier,
    digest_relation_sql_v2,
)
from increment.semantics._spillover import relation_names
from increment.semantics.models import Definitions

_PROJECTIONS = {
    "roster": ("person_id", "group_id", "stratum"),
    "offer_counts": ("group_id", "policy", "offers"),
    "group_policy": ("group_id", "policy"),
    "offered": ("person_id", "offered"),
    "outcomes": ("person_id", "outcome"),
}
_DESIGN_ROLES = ("roster", "offer_counts")
# str.strip() also treats these Unicode and control characters as whitespace;
# SQL TRIM's default set depends on the warehouse.
_LABEL_WHITESPACE = (
    "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    + "".join(chr(i) for i in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000"
)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    session: WarehouseSession
    tables: Mapping[str, Table]

    def table(self, role: str) -> Table:
        if role not in self.tables:
            code = "spillover.design.invalid" if role in _DESIGN_ROLES else "spillover.log.keys"
            refuse_input(code, role)
        return self.tables[role]

    def any_rows(self, rows: Table) -> bool:
        return bool(self.session.fetch_arrow(rows.limit(1).count()).as_py())


@contextmanager
def _capture(
    con: SQLBackend, names: Mapping[str, str], roles: tuple[str, ...]
) -> Iterator[_Snapshot]:
    _require_warehouse_digest_verifier(
        con,
        relation="roster",
        digest_format=SQL_DIGEST_FORMAT,
        route="use spillover.from_frame to analyze dataframes",
    )
    live = WarehouseSession(con, Definitions())
    inputs = {}
    for role in roles:
        *namespace, name = names[role].split(".")
        database = ".".join(namespace) or None
        try:
            table = con.table(name, database=database)
        except TableNotFound:
            break
        columns = _PROJECTIONS[role]
        if not set(columns).issubset(table.columns):
            break
        inputs[role] = table.select(*columns)
    # One CTAS captures every available stream, before validating even the first row.
    # A missing projection is reported only after earlier roles have been checked.
    with translating_compile_gaps():
        pinned = live.pin_sources((), extra_relations=inputs)
        try:
            yield _Snapshot(pinned, {role: pinned.source_sql(role) for role in inputs})
        finally:
            pinned.drop_materialized()


def _key_kind(table: Table, key: str) -> str | None:
    dtype = table.schema()[key]
    if dtype.is_integer():
        return "int64"
    if dtype.is_string():
        return "string"
    return None


def _invalid_label(table: Table, key: str):
    col = table[key]
    return col.isnull() | (col.translate(_LABEL_WHITESPACE, "") == "")


def _invalid_id(table: Table, key: str):
    return _invalid_label(table, key) if _key_kind(table, key) == "string" else table[key].isnull()


def _roster_groups(snapshot: _Snapshot, roster: Table) -> tuple[Table, int, int]:
    if not (
        _key_kind(roster, "person_id")
        and _key_kind(roster, "group_id")
        and roster.schema()["stratum"].is_string()
    ):
        refuse_input("spillover.design.invalid", "roster")
    invalid = (
        _invalid_id(roster, "person_id")
        | _invalid_id(roster, "group_id")
        | _invalid_label(roster, "stratum")
    )
    counts = snapshot.session.fetch_arrow(
        roster.aggregate(n=roster.count(), people=roster.person_id.nunique(), invalid=invalid.any())
    ).to_pylist()[0]
    if not counts["n"] or counts["invalid"]:
        refuse_input("spillover.design.invalid", "roster")
    if counts["n"] != counts["people"]:
        refuse_input("spillover.design.duplicate", "roster")
    groups = roster.group_by("group_id", "stratum").aggregate(size=roster.count())
    owners = snapshot.session.fetch_arrow(
        groups.aggregate(n=groups.count(), unique_ids=groups.group_id.nunique())
    ).to_pylist()[0]
    if owners["n"] != owners["unique_ids"]:
        refuse_input("spillover.design.duplicate", "roster")
    return groups, counts["people"], owners["n"]


def _offer_schedule(
    snapshot: _Snapshot, groups: Table, group_n: int, design: SpilloverDesign
) -> Table:
    counts = snapshot.table("offer_counts")
    if (
        _key_kind(counts, "group_id") != _key_kind(groups, "group_id")
        or not counts.schema()["policy"].is_string()
    ):
        refuse_input("spillover.design.invalid", "offer_counts")
    invalid_key = _invalid_id(counts, "group_id") | _invalid_label(counts, "policy")
    if snapshot.any_rows(counts.filter(invalid_key)):
        refuse_input("spillover.design.invalid", "offer_counts")
    pairs = counts.group_by("group_id", "policy").aggregate(copies=counts.count())
    summary = snapshot.session.fetch_arrow(
        pairs.aggregate(n=pairs.count(), duplicate=(pairs.copies > 1).any())
    ).to_pylist()[0]
    if summary["duplicate"]:
        refuse_input("spillover.design.duplicate", "offer_counts")
    if not counts.schema()["offers"].is_integer() or summary["n"] != group_n * len(design.policies):
        refuse_input("spillover.design.quota", "offer_counts")
    # A foreign group cannot disappear in the join. Unique valid pairs numbering G*P
    # then imply that every group has exactly one quota under every declared policy.
    schedule = counts.left_join(groups, "group_id").select(counts, groups.size, groups.stratum)
    invalid = (
        schedule.size.isnull()
        | schedule.policy.notin(design.policies)
        | schedule.offers.isnull()
        | (schedule.offers < 0)
        | (schedule.offers > schedule.size)
        | (
            (schedule.offers == schedule.size)
            & schedule.policy.isin(tuple(sorted(design.compared_policies)))
        )
    )
    if snapshot.any_rows(schedule.filter(invalid)):
        refuse_input("spillover.design.quota", "offer_counts")
    return schedule


def _check_strata(snapshot: _Snapshot, groups: Table, design: SpilloverDesign) -> None:
    expected = {s: sum(q.values()) for s, q in design.groups_per_policy.items()}
    if snapshot.any_rows(groups.filter(groups.stratum.notin(tuple(expected)))):
        refuse_input("spillover.design.quota", "groups_per_policy")
    sizes = groups.group_by("stratum").aggregate(n=groups.count())
    rows = snapshot.session.fetch_arrow(sizes).to_pylist()
    if {r["stratum"]: r["n"] for r in rows} != expected:
        refuse_input("spillover.design.quota", "groups_per_policy")


def _identity(snapshot: _Snapshot) -> str:
    digest = hashlib.sha256(b"increment.spillover.population\x00v1\x00")
    for role, key in (("roster", ("person_id",)), ("offer_counts", ("group_id", "policy"))):
        table = snapshot.table(role)
        schema = {name: _key_kind(table, name) for name in _PROJECTIONS[role]}
        relation = digest_relation_sql_v2(
            snapshot.session.con, table, f"spillover.{role}", schema, primary_key=key
        )
        digest.update(bytes.fromhex(relation.content_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _DesignTables:
    groups: Table
    schedule: Table
    people_n: int
    group_n: int
    digest: str


def _checked_design(snapshot: _Snapshot, design: SpilloverDesign) -> _DesignTables:
    groups, people_n, group_n = _roster_groups(snapshot, snapshot.table("roster"))
    schedule = _offer_schedule(snapshot, groups, group_n, design)
    _check_strata(snapshot, groups, design)
    return _DesignTables(groups, schedule, people_n, group_n, _identity(snapshot))


def _profiles(
    snapshot: _Snapshot, schedule: Table, policies: tuple[str, ...]
) -> tuple[GroupProfile, ...]:
    columns = {
        f"offers_{i}": schedule.offers.max(where=schedule.policy == p)
        for i, p in enumerate(policies)
    }
    wide = schedule.group_by("group_id", "stratum", "size").aggregate(**columns)
    return tuple(
        GroupProfile(
            group_id=r["group_id"],
            stratum=r["stratum"],
            size=r["size"],
            offers=tuple(r[k] for k in columns),
        )
        for r in snapshot.session.fetch_arrow(wide).to_pylist()
    )


def _log_keys(snapshot: _Snapshot, role: str, eligible: Table, key: str, size: int) -> Table:
    table = snapshot.table(role)
    if _key_kind(table, key) != _key_kind(eligible, key):
        refuse_input("spillover.log.keys", role)
    summary = snapshot.session.fetch_arrow(
        table.aggregate(n=table.count(), owners=table[key].nunique())
    ).to_pylist()[0]
    n = summary["n"]
    if n != summary["owners"] or n > size or (role != "outcomes" and n != size):
        refuse_input("spillover.log.keys", role)
    if snapshot.any_rows(table.anti_join(eligible, key)):
        refuse_input("spillover.log.keys", role)
    return table


def _assigned_groups(snapshot: _Snapshot, frozen: _DesignTables, design: SpilloverDesign) -> Table:
    policies = _log_keys(snapshot, "group_policy", frozen.groups, "group_id", frozen.group_n)
    if not policies.schema()["policy"].is_string() or snapshot.any_rows(
        policies.filter(policies.policy.isnull() | policies.policy.notin(design.policies))
    ):
        refuse_input("spillover.log.keys", "group_policy")
    assigned = frozen.schedule.semi_join(policies, ["group_id", "policy"])
    cells = assigned.group_by("stratum", "policy").aggregate(
        n=assigned.count(), roster_mass=assigned.size.sum()
    )
    rows = snapshot.session.fetch_arrow(cells).to_pylist()
    observed = {(r["stratum"], r["policy"]): r["n"] for r in rows}
    expected = {
        (s, p): n
        for s, allocation in design.groups_per_policy.items()
        for p, n in allocation.items()
    }
    if observed != expected or sum(r["roster_mass"] for r in rows) != frozen.people_n:
        refuse_input("spillover.log.quota", "group_policy")
    return assigned


def _check_binary(snapshot: _Snapshot, table: Table, role: str, column: str) -> None:
    dtype = table.schema()[column]
    flag = table[column]
    if dtype.is_boolean():
        if role == "outcomes":
            return
        bad = flag.isnull()
    elif dtype.is_integer():
        bad = flag.notin((0, 1))
        if role == "offered":
            bad = bad | flag.isnull()
    else:
        refuse_input("spillover.log.binary", role)
    if snapshot.any_rows(table.filter(bad)):
        refuse_input("spillover.log.binary", role)


def _indicator(table: Table, column: str):
    value = table[column]
    # PostgreSQL has no direct boolean-to-bigint cast.
    return value.ifelse(1, 0) if table.schema()[column].is_boolean() else value


def _untreated(snapshot: _Snapshot, frozen: _DesignTables, assigned: Table) -> Table:
    roster = snapshot.table("roster")
    log = _log_keys(snapshot, "offered", roster, "person_id", frozen.people_n)
    _check_binary(snapshot, log, "offered", "offered")
    people = roster.join(log, "person_id").select(
        roster.person_id, roster.group_id, flag=_indicator(log, "offered")
    )
    totals = people.group_by("group_id").aggregate(actual=people.flag.sum())
    checked = assigned.join(totals, "group_id")
    if snapshot.any_rows(checked.filter(checked.offers != checked.actual)):
        refuse_input("spillover.log.quota", "offered")
    return people.filter(people.flag == 0)


def _observed_cells(
    snapshot: _Snapshot, frozen: _DesignTables, design: SpilloverDesign
) -> tuple[PolicyCell, ...]:
    assigned = _assigned_groups(snapshot, frozen, design)
    untreated = _untreated(snapshot, frozen, assigned)
    outcomes = _log_keys(
        snapshot, "outcomes", snapshot.table("roster"), "person_id", frozen.people_n
    )
    reported = outcomes.filter(outcomes.outcome.notnull())
    if snapshot.any_rows(untreated.anti_join(reported, "person_id")):
        refuse_input("spillover.log.keys", "outcomes")
    _check_binary(snapshot, outcomes, "outcomes", "outcome")
    measured = untreated.join(outcomes, "person_id")
    by_group = measured.group_by("group_id").aggregate(
        unoffered=measured.count(), events=_indicator(measured, "outcome").sum()
    )
    compared = assigned.filter(assigned.policy.isin(tuple(sorted(design.compared_policies))))
    groups = compared.join(by_group, "group_id")
    groups = groups.mutate(total=groups.size.cast("float64") * groups.events / groups.unoffered)
    cells = groups.group_by("stratum", "policy").aggregate(
        groups=groups.count(),
        mean_total=groups.total.mean(),
        ss=groups.total.var(how="pop") * groups.count(),
        unoffered=groups.unoffered.sum(),
        events=groups.events.sum(),
    )
    rows = snapshot.session.fetch_arrow(cells).to_pylist()
    # Match the frame's label order regardless of the connection's collation.
    rows.sort(key=lambda r: (r["stratum"], r["policy"]))
    return tuple(PolicyCell(**row) for row in rows)


class WarehouseSpilloverSource:
    """Bind scoped table names on a caller-owned connection."""

    def __init__(self, con: SQLBackend, relations: Mapping[str, object]) -> None:
        self._con = con
        self._names = relation_names(relations)

    def freeze(self, design: SpilloverDesign) -> SpilloverPopulation:
        if not isinstance(design, SpilloverDesign):
            refuse_input("spillover.design.required", "design")
        with _capture(self._con, self._names, _DESIGN_ROLES) as snapshot:
            frozen = _checked_design(snapshot, design)
            groups = _profiles(snapshot, frozen.schedule, design.policies)
            return SpilloverPopulation(design=design, groups=groups, roster_digest=frozen.digest)

    def observe(self, population: SpilloverPopulation) -> tuple[PolicyCell, ...]:
        with _capture(self._con, self._names, tuple(_PROJECTIONS)) as snapshot:
            frozen = _checked_design(snapshot, population.design)
            if frozen.digest != population.roster_digest:
                refuse_input("spillover.roster.changed", "roster")
            return _observed_cells(snapshot, frozen, population.design)
