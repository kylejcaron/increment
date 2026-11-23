"""Capture the two randomizations and reduce untreated outcomes inside the warehouse."""

# Warehouse availability must be checked before importing optional dependencies.
# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4
from weakref import WeakKeyDictionary

from increment._optional import require_warehouse

require_warehouse("increment.query.spillover")

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import sqlglot as sg
from ibis import Table
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import SQLGlotCompiler
from ibis.common.exceptions import TableNotFound

from increment._compile_gaps import translating_compile_gaps
from increment._spillover_design import (
    GroupProfile,
    PolicyCell,
    SpilloverDesign,
    SpilloverPopulation,
    refuse_input,
)
from increment._spillover_design import _raise as _refuse_design
from increment.errors import CodedError, DefinitionError, InvalidRequestError, RefusalSpec, refuse
from increment.query._spillover_sql import (
    _bind_query,
    _qualified,
    _Query,
    duckdb_function_guard,
    duckdb_identifier_sql,
    view_table_parts,
)
from increment.query._spillover_view_sources import ViewSources, view_sources
from increment.query.artifact_ports import SQL_DIGEST_FORMAT
from increment.query.builders import (
    _local_date,
    _naive_utc_ts,
    _scope_exposure_events,
    declared_exposure_rows,
    measure_rows,
    metric_events,
    outcome_rows,
)
from increment.query.fact_resolution import (
    _dim_joined_fact_table,
    _find_fact_source,
    _raise_definition,
    null_value_count,
    refuse_null_fact_values,
    resolve_shared_roster,
    shared_roster_parts,
    validate_shared_fact,
    validate_shared_roster,
    validate_sql_exposure_columns,
)
from increment.query.session import (
    SourceSnapshotEvidence,
    WarehouseSession,
    _require_warehouse_digest_verifier,
    digest_relation_sql_v2,
)
from increment.semantics.loader import verify_sql_admission_matches_execution
from increment.semantics.models import (
    ConversionMetric,
    Definitions,
    Experiment,
    Exposure,
    Fact,
    FactSource,
    SpilloverDeclaration,
)

if TYPE_CHECKING:
    from increment.semantics.shared import BoundSharedMetric

_INCOMPLETE_WINDOW = RefusalSpec(
    "spillover.window.incomplete",
    InvalidRequestError,
    template=(
        "The full {window_days}-day window for {metric!r} is not observed ({field}). "
        "Check the observation end and use a completed load receipt for every contributing feed."
    ),
)

_PROJECTIONS = {
    "roster": ("person_id", "group_id", "stratum"),
    "offer_counts": ("group_id", "policy", "offers"),
    "group_policy": ("group_id", "policy"),
}
_DESIGN_ROLES = ("roster", "offer_counts")
_SHARED_ROSTER = "shared_roster"
# str.strip() also treats these Unicode and control characters as whitespace;
# SQL TRIM's default set depends on the warehouse.
_LABEL_WHITESPACE = (
    "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    + "".join(chr(i) for i in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000"
)
_Locator = tuple[str, str, str]
_DUCKDB_HANDLES: WeakKeyDictionary[Any, str] = WeakKeyDictionary()


def _duckdb_catalogs(con: SQLBackend, catalogs: set[str]) -> Table:
    """Identify a catalog by its file location, never by hashing its contents."""
    # A relative path reported by older DuckDB versions is ambiguous outside
    # this attachment, just like an unnamed in-memory database.
    metadata = con.sql(
        rf"""
        SELECT {duckdb_identifier_sql("database_name")} AS catalog, database_oid,
               CASE WHEN system.main.regexp_matches(
                   path, '^(/|[A-Za-z]:[/\\]|\\\\|[A-Za-z][-+.A-Za-z0-9]*://)'
               ) THEN system.main.sha256(path) END AS file_digest
        FROM system.main.duckdb_databases() WHERE NOT internal
        """,
        schema={"catalog": "string", "database_oid": "int64", "file_digest": "string"},
    )
    connection = cast(Any, con).con
    # A statement logger may wrap execute() without changing the underlying DB-API session.
    handle = getattr(connection.cursor, "__self__", connection)
    token = _DUCKDB_HANDLES.get(handle)
    if token is None:
        token = _DUCKDB_HANDLES.setdefault(handle, uuid4().hex)
    selected = metadata.filter(metadata.catalog.isin(tuple(sorted(catalogs))))
    return selected.select(
        selected.catalog,
        location=selected.file_digest.fill_null(token + ":" + selected.database_oid.cast("string")),
    )


def _source_checks(
    con: SQLBackend,
    names: Mapping[str, _Locator],
    locations: Mapping[str, str],
    views: ViewSources | None,
) -> Table | None:
    """One status row to capture in the same statement as the named inputs."""
    if views is not None:
        return views.checks(con, names, _duckdb_catalogs(con, views.catalogs), locations)
    if con.name != "snowflake":
        return None
    checks = []
    for role, (catalog, schema, name) in names.items():
        tables = sg.table("TABLES", db="INFORMATION_SCHEMA", catalog=catalog, quoted=True)
        # Quoted identifiers can resolve exactly or in upper case, depending
        # on session settings. Accept only one visible, non-temporary object.
        query = (
            sg.select("COUNT(*) = 1 AND MIN(IS_TEMPORARY) = 'NO'")
            .from_(tables)
            .where(
                sg.column("TABLE_SCHEMA").isin(schema, schema.upper()),
                sg.column("TABLE_NAME").isin(name, name.upper()),
            )
        )
        checks.append(query.subquery().as_(role, quoted=True))
    return con.sql(
        sg.select(*checks).sql(dialect=con.dialect), schema=dict.fromkeys(names, "boolean")
    )


class _SpilloverExactText(ops.Value):
    """Keep identifiers byte-distinct through SQL joins, grouping and uniqueness checks."""

    arg: ops.Value[dt.String]
    dtype = dt.string
    shape = rlz.shape_like("arg")


def _exact_text_sql(compiler: Any, _op: _SpilloverExactText, *, arg: Any):
    dialect = compiler.dialect
    if dialect == "duckdb" or dialect == "postgres":
        collation = sg.to_identifier("binary" if dialect == "duckdb" else "C", quoted=True)
    else:
        # BigQuery and Snowflake: remove any inherited case/accent-insensitive collation.
        collation = sg.exp.Literal.string("")
    return sg.exp.Collate(this=arg, expression=collation)


SQLGlotCompiler.visit__SpilloverExactText = _exact_text_sql  # ty: ignore[unresolved-attribute]


def _exact_keys(
    table: Table, keys: tuple[str, ...] = ("person_id", "group_id", "stratum", "policy")
) -> Table:
    return table.mutate(
        **{
            name: _SpilloverExactText(table[name].op()).to_expr()
            for name, dtype in table.schema().items()
            if name in keys and dtype.is_string()
        }
    )


@dataclass(frozen=True, slots=True)
class _Snapshot:
    session: WarehouseSession
    tables: Mapping[str, Table]
    unbound: frozenset[str]
    unavailable: Mapping[str, Exception]

    def table(self, role: str, *, field: str | None = None) -> Table:
        field = field or role
        if role in self.unbound:
            if self.session.con.name == "duckdb":
                refuse_input("spillover.roster.changed", "roster")
            _refuse_design(
                "spillover.design.invalid",
                field=field,
                requirement="use an unambiguous, non-temporary table or view in this session",
            )
        if role in self.unavailable:
            cause = self.unavailable[role]
            if isinstance(cause, CodedError) or role in _PROJECTIONS:
                raise cause
            try:
                if field == "offered":
                    refuse_input("spillover.log.keys", field)
                _raise_definition("definition.fact_source.sql_unresolved", fact_source=field)
            except (DefinitionError, InvalidRequestError) as exc:
                raise exc from cause
        if role not in self.tables:
            code = "spillover.design.invalid" if role in _DESIGN_ROLES else "spillover.log.keys"
            refuse_input(code, field)
        return self.tables[role]

    def scalar(self, expression: ir.Scalar) -> Any:
        return self.session.fetch_arrow(expression).as_py()

    def any_rows(self, rows: Table) -> bool:
        return bool(self.scalar(rows.limit(1).count()))


@contextmanager
def _capture(
    live: WarehouseSession,
    names: Mapping[str, _Locator],
    roles: tuple[str, ...],
    locations: Mapping[str, str],
    views: ViewSources | None,
    queries: tuple[_Query, ...] = (),
    shared: BoundSharedMetric | None = None,
) -> Iterator[_Snapshot]:
    con = live.con
    inputs = {}
    unavailable: dict[str, Exception] = {}
    for role in roles:
        catalog, schema, name = names[role]
        try:
            table = con.table(name, database=(catalog, schema))
        except TableNotFound:
            break
        except Exception as exc:
            if views is None:
                raise
            unavailable[role] = exc
            break
        columns = _PROJECTIONS[role]
        if not set(columns).issubset(table.columns):
            break
        inputs[role] = table.select(*columns)
    targets = {r: names[r] for r in (roles if con.name == "duckdb" else tuple(inputs))}
    if shared is not None and shared.roster is not None:
        location = names[_SHARED_ROSTER]
        try:
            inputs[_SHARED_ROSTER] = resolve_shared_roster(
                con, shared.roster, metric=shared.name, location=location
            )
        except DefinitionError as exc:
            unavailable[_SHARED_ROSTER] = exc
        if _SHARED_ROSTER in inputs or con.name == "duckdb":
            targets[_SHARED_ROSTER] = location
    dependencies: dict[str, tuple[str, ...]] = {key: (key,) for key in targets}
    function_checks = {}
    for index, query in enumerate(queries):
        keys = tuple(f"q{index}_{i}" for i in range(len(query.dependencies)))
        targets.update(zip(keys, query.dependencies, strict=True))
        if query.function_guard:
            flag = f"q{index}_functions"
            guarded = con.sql(
                f"SELECT ({query.function_guard}) AS valid", schema={"valid": "boolean"}
            )
            function_checks[flag] = guarded.valid.as_scalar()
            keys = (*keys, flag)
        dependencies[query.sql] = keys
        try:
            inputs[query.sql] = live.source_sql(query.native_sql)
        except Exception as exc:
            # A detached catalog can make even schema inference fail. The
            # captured binding checks below take precedence over a missing log.
            unavailable[query.sql] = exc
    checks = _source_checks(con, targets, locations, views) if targets else None
    if checks is not None and function_checks:
        checks = checks.mutate(**function_checks)
    streams = dict(inputs)
    if checks is not None:
        for role, table in inputs.items():
            streams[role] = table.filter(
                [cast(ir.BooleanScalar, checks[key].as_scalar()) for key in dependencies[role]]
            )
        streams["source_checks"] = checks
    with translating_compile_gaps():
        pinned = live.pin_sources((), extra_relations=streams)
        try:
            unbound = frozenset()
            if checks is not None:
                flags = pinned.fetch_arrow(pinned.source_sql("source_checks")).to_pylist()[0]
                unbound = frozenset(
                    role for role, keys in dependencies.items() if any(not flags[k] for k in keys)
                )
            tables = {
                role: _exact_keys(pinned.source_sql(role))
                if role in names
                else pinned.source_sql(role)
                for role in inputs
            }
            yield _Snapshot(pinned, tables, unbound, unavailable)
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


def _check_log_keys(
    snapshot: _Snapshot, table: Table, role: str, eligible: Table, key: str, size: int
) -> None:
    if _key_kind(table, key) != _key_kind(eligible, key):
        refuse_input("spillover.log.keys", role)
    summary = snapshot.session.fetch_arrow(
        table.aggregate(n=table.count(), owners=table[key].nunique())
    ).to_pylist()[0]
    n = summary["n"]
    if n != summary["owners"] or n != size:
        refuse_input("spillover.log.keys", role)
    if snapshot.any_rows(table.anti_join(eligible, key)):
        refuse_input("spillover.log.keys", role)


def _assigned_groups(snapshot: _Snapshot, frozen: _DesignTables, design: SpilloverDesign) -> Table:
    policies = snapshot.table("group_policy")
    _check_log_keys(snapshot, policies, "group_policy", frozen.groups, "group_id", frozen.group_n)
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


def _fact_table(snapshot: _Snapshot, source: FactSource, unit: str) -> Table:
    dims = {dim.sql: dim for dim in snapshot.session.defs.dim_sources if dim.name in source.dims}

    def resolve(sql: str) -> Table:
        dim = dims.get(sql)
        field = source.name if sql == source.sql or dim is None else dim.name
        return _exact_keys(snapshot.table(sql, field=field), (*source.entities, "unit_id"))

    return _dim_joined_fact_table(
        snapshot.session.con,
        snapshot.session.defs,
        source,
        unit,
        {},
        resolve_sql=resolve,
        naive_utc=True,
        execute=snapshot.scalar,
    )


def _assignment_rows(
    snapshot: _Snapshot, experiment: Experiment, exposure: Exposure, source: FactSource | None
) -> Table:
    if exposure.sql is not None:
        raw = snapshot.table(exposure.sql, field="offered")
        validate_sql_exposure_columns(raw, experiment.name, exposure.name)
        keyed = "experiment_id" in raw.columns
        raw = _naive_utc_ts(raw)
    else:
        assert exposure.fact is not None and source is not None
        keyed = "experiment_id" in snapshot.table(source.sql, field="offered").columns
        raw = _fact_table(snapshot, source, experiment.unit)
        raw = raw.filter(raw.event == exposure.fact)
    if (
        "group_id" not in raw.columns
        or not raw.group_id.type().is_string()
        or not raw.ts.type().is_timestamp()
    ):
        refuse_input("spillover.log.keys", "offered")
    raw = _exact_keys(raw, ("unit_id", "experiment_id", "group_id"))
    scoped = declared_exposure_rows(raw, exposure, experiment, keyed=keyed)
    # An undated draw cannot be ruled out of enrollment, even if the same
    # person has another draw with a valid timestamp.
    if snapshot.any_rows(scoped.filter(scoped.ts.isnull())):
        refuse_input("spillover.log.keys", "offered")
    scoped = _scope_exposure_events(scoped, experiment)
    return scoped.select(
        person_id=scoped.unit_id, label=scoped.group_id, first_exposure_ts=scoped.ts
    )


def _check_window(
    snapshot: _Snapshot,
    anchors: Table,
    experiment: Experiment,
    metric: ConversionMetric,
    source: FactSource,
) -> None:
    days = metric.window_days
    assert days is not None
    receipt = snapshot.session.source_snapshot_evidence
    if receipt is None:
        refuse(
            _INCOMPLETE_WINDOW,
            field="source_snapshot_evidence",
            metric=metric.name,
            window_days=days,
        )
    offset = experiment.day_boundary_offset
    # A complete day requires midnight following that day, not its first event.
    edges = [("observation_cutoff_ts", (receipt.observation_cutoff_ts + offset).date().toordinal())]
    for feed in (source.name, *source.dims):
        watermark = receipt.complete_through_by_feed.get(feed)
        if watermark is None:
            refuse(_INCOMPLETE_WINDOW, field=feed, metric=metric.name, window_days=days)
        edges.append((feed, (watermark + offset).date().toordinal()))
    horizon = experiment.observation_horizon_day
    if horizon is not None:
        edges.append(("observation_end", horizon.toordinal() + 1))
    field, edge = min(edges, key=lambda item: item[1])
    latest_anchor = edge - days
    day = _local_date(anchors.first_exposure_ts, experiment)
    if latest_anchor < 1 or snapshot.any_rows(
        anchors.filter(day > date.fromordinal(latest_anchor))
    ):
        refuse(_INCOMPLETE_WINDOW, field=field, metric=metric.name, window_days=days)


def _nonoffer_anchors(
    snapshot: _Snapshot,
    frozen: _DesignTables,
    assigned: Table,
    log: Table,
    experiment: Experiment,
    offered_group: str,
) -> Table:
    roster = snapshot.table("roster")
    _check_log_keys(snapshot, log, "offered", roster, "person_id", frozen.people_n)
    invalid = log.label.isnull() | log.label.notin((experiment.control_group, offered_group))
    if snapshot.any_rows(log.filter(invalid)):
        refuse_input("spillover.log.keys", "offered")
    people = roster.join(log, "person_id").select(
        unit_id=roster.person_id,
        group_id=roster.group_id,
        first_exposure_ts=log.first_exposure_ts,
        flag=(log.label == offered_group).ifelse(1, 0),
    )
    totals = people.group_by("group_id").aggregate(actual=people.flag.sum())
    checked = assigned.join(totals, "group_id")
    if snapshot.any_rows(checked.filter(checked.offers != checked.actual)):
        refuse_input("spillover.log.quota", "offered")
    return people.filter(people.flag == 0)


def _validate_shared_conversion(
    snapshot: _Snapshot, binding: BoundSharedMetric, source: FactSource
) -> None:
    """Check the portable conversion's event grain and its full eligible population."""
    (fact,) = binding.facts
    assert binding.roster is not None
    relation = snapshot.table(source.sql, field=source.name)
    validate_shared_fact(
        snapshot.scalar,
        _exact_keys(relation, (fact.row_key,)),
        source,
        fact,
        metric=binding.name,
        unit=binding.unit,
        captured=True,
    )
    roster = snapshot.table("roster")
    validate_shared_roster(
        snapshot.scalar,
        roster.select(unit_id=roster.person_id),
        _exact_keys(snapshot.table(_SHARED_ROSTER, field=binding.roster.table), ("unit_id",)),
        roster=binding.roster,
        metric=binding.name,
    )


def _group_conversions(
    snapshot: _Snapshot,
    anchors: Table,
    experiment: Experiment,
    metric: ConversionMetric,
    source: FactSource,
    fact: Fact,
) -> Table:
    rows = _fact_table(snapshot, source, experiment.unit)
    if (
        _key_kind(rows, "unit_id") != _key_kind(anchors, "unit_id")
        or not rows.ts.type().is_timestamp()
    ):
        refuse_input("spillover.design.invalid", source.name)
    rows = _exact_keys(rows, ("unit_id",))
    band = (0, metric.window_days)
    if fact.column is not None and fact.missing == "error":
        used = outcome_rows(measure_rows(rows, metric), anchors, experiment, band=band)
        count = null_value_count(fact, used)
        n = snapshot.session.fetch_arrow(count.null_rows).as_py()
        refuse_null_fact_values({"n": count}, {"n": n})
    events = metric_events(rows, metric, value_column=fact.column, null_policy=fact.missing)
    conversions = outcome_rows(events, anchors, experiment, band=band).select("unit_id").distinct()
    measured = anchors.left_join(conversions, "unit_id").select(
        anchors.group_id, bought=conversions.unit_id.notnull().ifelse(1, 0)
    )
    return measured.group_by("group_id").aggregate(
        unoffered=measured.count(), events=measured.bought.sum()
    )


def _observed_cells(
    snapshot: _Snapshot, assigned: Table, by_group: Table, design: SpilloverDesign
) -> tuple[PolicyCell, ...]:
    compared = assigned.filter(assigned.policy.isin(tuple(sorted(design.compared_policies))))
    groups = compared.join(by_group, "group_id")
    groups = groups.mutate(total=groups.size.cast("float64") * groups.events / groups.unoffered)
    # Recenter before averaging: even a repeated fractional total can drift
    # when summed thousands of times without subtracting a shared origin.
    centered = groups.group_by("stratum", "policy").mutate(origin=groups.total.mean())
    delta = centered.total - centered.origin
    cells = centered.group_by("stratum", "policy").aggregate(
        groups=centered.count(),
        mean_total=centered.origin.max() + delta.mean(),
        ss=delta.var(how="pop") * centered.count(),
        unoffered=centered.unoffered.sum(),
        events=centered.events.sum(),
    )
    rows = snapshot.session.fetch_arrow(cells).to_pylist()
    # Match the frame's label order regardless of the connection's collation.
    rows.sort(key=lambda r: (r["stratum"], r["policy"]))
    return tuple(PolicyCell(**row) for row in rows)


def _selected_fact(
    definitions: Definitions, name: str, properties: set[str]
) -> tuple[FactSource, Fact]:
    source, fact = _find_fact_source(definitions, name)
    # A valid static or temporal dim LEFT-joins at most one row. If no selected
    # column comes from it, omitting that join preserves every event and null.
    used = {
        d.name for d in definitions.dim_sources if any(p.name in properties for p in d.properties)
    }
    scoped = source.model_copy(
        update={"facts": (fact,), "dims": tuple(dim for dim in source.dims if dim in used)}
    )
    return scoped, fact


def _trial_definitions(
    definitions: Definitions,
    experiment: Experiment,
    metric: ConversionMetric,
    exposure: Exposure,
    reads: tuple[FactSource, ...],
) -> Definitions:
    # One physical stream can serve both roles. Capture its union once, but
    # join and certify only the dimensions each role actually reads.
    facts = {fact.name for source in reads for fact in source.facts}
    joins = {(source.name, dim) for source in reads for dim in source.dims}
    sources = tuple(
        fs.model_copy(
            update={
                "facts": tuple(f for f in fs.facts if f.name in facts),
                "dims": tuple(d for d in fs.dims if (fs.name, d) in joins),
            }
        )
        for fs in sorted(definitions.fact_sources, key=lambda fs: fs.name)
        if any(f.name in facts for f in fs.facts)
    )
    names = {dim for _, dim in joins}
    dims = tuple(
        d for d in sorted(definitions.dim_sources, key=lambda d: d.name) if d.name in names
    )
    return definitions.model_copy(
        update={
            "fact_sources": sources,
            "dim_sources": dims,
            "experiments": (experiment,),
            "metrics": (metric,),
            "exposures": (exposure,),
        }
    )


class WarehouseSpilloverSource:
    """Bind scoped table names on a caller-owned connection."""

    def __init__(
        self,
        con: SQLBackend,
        *,
        definitions: Definitions,
        experiment: Experiment,
        metric: ConversionMetric,
        source_snapshot_evidence: SourceSnapshotEvidence | None = None,
    ) -> None:
        verify_sql_admission_matches_execution(definitions, con)
        _require_warehouse_digest_verifier(
            con,
            relation="roster",
            digest_format=SQL_DIGEST_FORMAT,
            route="use from_frame with shape='summary' to analyze dataframes",
        )
        design = experiment.design
        if not isinstance(design, SpilloverDeclaration):
            refuse_input("spillover.design.required", "design")
        self._experiment = experiment
        self._offered_group = design.offered_group
        self._metric = metric
        self._exposure = next(e for e in definitions.exposures if e.name == experiment.exposure)
        self._source, self._fact = _selected_fact(
            definitions, metric.fact, {flt.property for flt in metric.filters}
        )
        self._assignment_source: FactSource | None = None
        if self._exposure.fact is not None:
            self._assignment_source, _ = _selected_fact(
                definitions,
                self._exposure.fact,
                {"group_id", *(flt.property for flt in self._exposure.filters)},
            )
        selected = _trial_definitions(
            definitions,
            experiment,
            metric,
            self._exposure,
            (self._source, self._assignment_source) if self._assignment_source else (self._source,),
        )
        shared = definitions._shared_metrics
        self._shared = (
            next((b for b in shared.metrics if b.name == metric.name), None) if shared else None
        )
        # Queries supplied to the live session below have already been translated.
        self._session = WarehouseSession(
            con,
            selected.model_copy(update={"dialect": con.name}),
            source_snapshot_evidence=source_snapshot_evidence,
        )
        self._names = {
            role: _qualified(con, tuple(name.split(".")))
            for role, name in design.table_names().items()
        }
        if self._shared is not None and self._shared.roster is not None:
            self._names[_SHARED_ROSTER] = _qualified(
                con, shared_roster_parts(self._shared.roster, metric=metric.name)
            )
        declarations = [(s.name, s.sql) for s in (*selected.fact_sources, *selected.dim_sources)]
        if self._exposure.sql is not None:
            declarations.append((self._exposure.name, self._exposure.sql))
        self._queries = tuple(
            _bind_query(con, field, sql, definitions.dialect) for field, sql in sorted(declarations)
        )
        semantics = {
            "experiment": experiment.model_dump(mode="json", exclude={"design"}),
            "day_boundary": experiment.day_boundary,
            "offered_group": design.offered_group,
            "metric": metric.model_dump(mode="json"),
            "exposure": self._exposure.model_dump(mode="json"),
            "sources": [s.model_dump(mode="json") for s in selected.fact_sources],
            "dimensions": [d.model_dump(mode="json") for d in selected.dim_sources],
        }
        if self._shared is not None:
            semantics["shared_conversion"] = asdict(self._shared)
        payload = json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()
        rows = [
            ("source", "warehouse"),
            ("backend", con.name),
            ("measurement", hashlib.sha256(payload).hexdigest()),
        ]
        for role, (catalog, schema, name) in sorted(self._names.items()):
            quoted = sg.table(name, db=schema, catalog=catalog, quoted=True)
            rows.append((role, quoted.sql(dialect=con.name)))
        rows.extend(
            (f"sql:{q.field}", hashlib.sha256(q.native_sql.encode()).hexdigest())
            for q in self._queries
        )
        self._locations: dict[str, str] = {}
        self._views: ViewSources | None = None
        if con.name == "duckdb":
            roots = set(self._names.values())
            roots.update(loc for q in self._queries for loc in q.dependencies)
            self._views = view_sources(con, roots, view_table_parts, duckdb_function_guard)
            catalogs = self._views.catalogs
            rows.extend(self._views.recipe)
            locations = con.to_pyarrow(_duckdb_catalogs(con, catalogs)).to_pylist()
            self._locations = {row["catalog"]: row["location"] for row in locations}
            if self._locations.keys() != catalogs:
                _refuse_design(
                    "spillover.design.invalid",
                    field="relations",
                    requirement="attach each named database before registering its tables",
                )
            rows.extend((f"catalog:{k}", v) for k, v in sorted(self._locations.items()))
        self.recipe: tuple[tuple[str, str], ...] = tuple(rows)

    def freeze(self, law: SpilloverDesign) -> SpilloverPopulation:
        if not isinstance(law, SpilloverDesign):
            refuse_input("spillover.design.required", "design")
        with _capture(
            self._session, self._names, _DESIGN_ROLES, self._locations, self._views
        ) as snapshot:
            frozen = _checked_design(snapshot, law)
            groups = _profiles(snapshot, frozen.schedule, law.policies)
            return SpilloverPopulation(design=law, groups=groups, roster_digest=frozen.digest)

    def observe(self, population: SpilloverPopulation) -> tuple[PolicyCell, ...]:
        with _capture(
            self._session,
            self._names,
            tuple(_PROJECTIONS),
            self._locations,
            self._views,
            self._queries,
            self._shared,
        ) as snapshot:
            frozen = _checked_design(snapshot, population.design)
            if frozen.digest != population.roster_digest:
                refuse_input("spillover.roster.changed", "roster")
            assigned = _assigned_groups(snapshot, frozen, population.design)
            log = _assignment_rows(
                snapshot, self._experiment, self._exposure, self._assignment_source
            )
            anchors = _nonoffer_anchors(
                snapshot, frozen, assigned, log, self._experiment, self._offered_group
            )
            _check_window(snapshot, log, self._experiment, self._metric, self._source)
            if self._shared is not None:
                _validate_shared_conversion(snapshot, self._shared, self._source)
            counts = _group_conversions(
                snapshot, anchors, self._experiment, self._metric, self._source, self._fact
            )
            return _observed_cells(snapshot, assigned, counts, population.design)
