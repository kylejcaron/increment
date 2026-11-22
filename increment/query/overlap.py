"""Capture aggregate evidence for two overlapping experiments from Ibis tables."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import file_digest, sha256
from math import isfinite
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from typing import TYPE_CHECKING, Any, NoReturn, cast

import ibis
import ibis.expr.operations as ops
import sqlglot
from duckdb import DuckDBPyConnection
from duckdb import Error as DuckDBError
from ibis.backends.duckdb import Backend
from ibis.common.exceptions import IbisError, TableNotFound
from ibis.expr import datatypes as dt
from ibis.expr import types as ir
from sqlglot import exp
from sqlglot.errors import SqlglotError

from increment._assignment_refusals import refuse_assignment_invalid
from increment._canonical import canonical_json_bytes
from increment._overlap_evidence import (
    CELL_IDS,
    DECISION_STATUSES,
    DELIVERY_KEYS,
    FACTOR_IDS,
    FEED_ROLES,
    GATE_KEYS,
    MAX_EXACT_INTEGER,
    OVERLAP_RELATIONS,
    PREASSIGNED_GATE_COLUMNS,
    OverlapColumn,
    overlap_binary_moments,
)
from increment._overlap_refusals import (
    CAPTURE_INVALID,
    EVIDENCE_INVALID,
    NUMERIC_RANGE,
    SCAN_INVALID,
)
from increment.errors import refuse
from increment.semantics.interaction_scan import (
    CombinationCell,
    CombinationFeed,
    CombinationRegistration,
    FocalSourceFacts,
    InteractionCombination,
    SourceFact,
    StudyFeedFact,
)
from increment.semantics.overlap import (
    OverlapCapture,
    OverlapCell,
    OverlapDeliveryCount,
    OverlapEvidence,
    OverlapPlan,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from increment._overlap_evidence import CellId, FactorId, FeedRole
    from increment.semantics.interaction_scan import InteractionScanPlan
    from increment.semantics.overlap import OverlapFactor, OverlapMetric


__all__ = ["bind_overlap_warehouse", "capture_overlap_warehouse"]


_KEYS = ("study_id", "unit_id")
_ASCII_FOLD = str.maketrans(ascii_uppercase, ascii_lowercase)
_CATALOG = f"""
SELECT d.path, d.database_name, t.schema_name, t.table_name, t.sql,
       c.column_name, c.data_type, c.column_default IS NOT NULL
FROM system.main.duckdb_databases() AS d
JOIN system.main.duckdb_tables() AS t USING (database_oid)
LEFT JOIN system.main.duckdb_columns() AS c
  ON c.table_oid = t.table_oid AND c.database_oid = d.database_oid
 AND c.schema_oid = t.schema_oid AND list_contains(?, c.column_name)
WHERE translate(d.database_name, '{ascii_uppercase}', '{ascii_lowercase}') = ?
  AND translate(t.schema_name, '{ascii_uppercase}', '{ascii_lowercase}') = ?
  AND translate(t.table_name, '{ascii_uppercase}', '{ascii_lowercase}') = ?
  AND d.type = 'duckdb' AND NOT d.internal AND NOT t.internal AND NOT t.temporary
"""


@ibis.udf.scalar.builtin(name="sha256", catalog="system", database="main")
def _sha256(value: str) -> str: ...  # ty: ignore[empty-body]


@ibis.udf.scalar.builtin(name="isfinite", catalog="system", database="main")
def _finite_timestamp(value: datetime) -> bool: ...  # ty: ignore[empty-body]


@ibis.udf.agg.builtin(name="favg", catalog="system", database="main")
def _compensated_mean(value: float) -> float: ...  # ty: ignore[empty-body]


@ibis.udf.agg.builtin(name="fsum", catalog="system", database="main")
def _compensated_sum(value: float) -> float: ...  # ty: ignore[empty-body]


def _unbound() -> NoReturn:
    refuse(CAPTURE_INVALID, role="snapshot", reason="unbound_generation")


def _evidence_error(role: str, reason: str, count: int) -> NoReturn:
    refuse(EVIDENCE_INVALID, role=role, reason=reason, count=count)


def _validate_declarations(plan: OverlapPlan, capture: OverlapCapture) -> None:
    if not isinstance(capture, OverlapCapture):
        refuse(CAPTURE_INVALID, role="snapshot", reason="invalid_value")
    capture.validate_for(plan)


def _physical_columns(plan: OverlapPlan, role: FeedRole) -> tuple[OverlapColumn, ...]:
    columns = OVERLAP_RELATIONS[role].columns
    if role == "roster" and plan.law.gate_basis == "preassigned_roster":
        return (*columns, *PREASSIGNED_GATE_COLUMNS)
    return columns


def _valid_type(kind: str, dtype: dt.DataType) -> bool:
    if kind == "timestamp":
        return isinstance(dtype, dt.Timestamp) and dtype.timezone == "UTC" and dtype.scale == 6
    if kind == "float":
        return dtype.is_floating()
    if kind == "integer":
        return dtype.is_integer()
    if kind == "boolean":
        return dtype.is_boolean()
    return dtype.is_string()


@dataclass(frozen=True)
class _Sources:
    backend: Backend
    catalog: str
    tables: Mapping[FeedRole, ir.Table]
    columns: Mapping[FeedRole, tuple[OverlapColumn, ...]]

    def scoped(self, plan: OverlapPlan) -> dict[FeedRole, ir.Table]:
        scoped = {}
        for role, table in self.tables.items():
            projection = table.filter(table.study_id == plan.study_id).select(
                *[column.name for column in self.columns[role]]
            )
            if role == "outcomes":
                projection = projection.mutate(value=projection.value.cast("float64"))
            scoped[role] = projection
        return scoped


def _sources(plan: OverlapPlan, tables: dict[FeedRole, ir.Table]) -> _Sources:
    backend = catalog = None
    columns = {role: _physical_columns(plan, role) for role in FEED_ROLES}
    for table in tables.values():
        if not isinstance(table, ir.Table):
            _unbound()
        op = table.op()
        if not isinstance(op, ops.DatabaseTable) or not isinstance(op.source, Backend):
            _unbound()
        if not op.namespace.catalog or not op.namespace.database:
            _unbound()
        folded_catalog = op.namespace.catalog.translate(_ASCII_FOLD)
        if backend is not None and (backend is not op.source or catalog != folded_catalog):
            _unbound()
        backend, catalog = op.source, folded_catalog
    assert backend is not None and catalog is not None
    return _Sources(backend, catalog, tables, columns)


@contextmanager
def _snapshot(sources: _Sources) -> Iterator[DuckDBPyConnection]:
    # A dedicated connection avoids joining or closing a transaction owned by the caller.
    with sources.backend.con.cursor() as cursor:
        cursor.execute("SET search_path = 'system.main'")
        cursor.execute("SET TimeZone = 'UTC'")
        cursor.execute("SET default_collation = ''")
        cursor.execute("BEGIN TRANSACTION")
        yield cursor
        cursor.execute("ROLLBACK")


def _identity(domain: str, parts: object) -> str:
    return sha256(domain.encode() + b"\0" + canonical_json_bytes(parts)).hexdigest()


def _locators(cursor: DuckDBPyConnection, sources: _Sources) -> dict[FeedRole, str]:
    source_ids: dict[FeedRole, str] = {}
    for role, table in sources.tables.items():
        op = cast("ops.DatabaseTable", table.op())
        namespace = op.namespace
        required = sources.columns[role]
        rows = cursor.execute(
            _CATALOG,
            [
                [col.name for col in required],
                sources.catalog,
                cast("str", namespace.database).translate(_ASCII_FOLD),
                op.name.translate(_ASCII_FOLD),
            ],
        ).fetchall()
        if not rows:
            _unbound()
        path, catalog, schema, name, ddl, *_ = rows[0]
        if not path or not Path(path).is_absolute() or not Path(path).is_file():
            _unbound()
        # Generated columns share `column_default` with stored defaults in the catalog.
        # Refuse both rather than execute an expression whose volatility is unknown.
        if any(row[7] for row in rows) or not _uncollated_table(ddl):
            _unbound()
        actual = {column: dtype for _, _, _, _, _, column, dtype, _ in rows if column is not None}
        _validate_columns(sources, role, table, actual)
        source_ids[role] = "duckdb-table-v1:" + _identity(
            "increment.overlap.duckdb.table.v1",
            (str(Path(path).resolve()), catalog, schema, name),
        )
    return source_ids


def _uncollated_table(ddl: str) -> bool:
    parsed = sqlglot.parse_one(ddl, dialect="duckdb")
    return (
        isinstance(parsed, exp.Create)
        and parsed.kind == "TABLE"
        and parsed.find(exp.CollateColumnConstraint) is None
    )


def _validate_columns(
    sources: _Sources, role: FeedRole, table: ir.Table, actual: Mapping[str, str]
) -> None:
    expected = sources.columns[role]
    absent = sum(column.name not in actual for column in expected)
    if absent:
        _evidence_error(role, "missing_columns", absent)
    schema = table.schema()
    mapper = sources.backend.compiler.type_mapper
    bad = sum(
        column.name not in schema
        or not _valid_type(column.kind, schema[column.name])
        or not _valid_type(column.kind, mapper.from_string(actual[column.name]))
        for column in expected
    )
    if bad:
        _evidence_error(role, "invalid_column_type", bad)


def _feed_digest(table: ir.Table, role: FeedRole) -> ir.Table:
    # Fixed-order typed row encoding; timestamps render in UTC in the owned connection.
    # Sorted SHA-256 buckets retain multiplicity without transferring row hashes.
    token = ibis.struct({name: table[name] for name in table.columns}).cast("json").cast("string")
    hashes = table.select(h=_sha256(token))
    buckets = hashes.mutate(bucket=hashes.h.substr(0, 3))
    segments = buckets.group_by(buckets.bucket).aggregate(
        h=_sha256(buckets.h.group_concat("", order_by=buckets.h))
    )
    ordered = segments.h.group_concat("", order_by=segments.bucket).fill_null("")
    return segments.aggregate(cast("ir.StringScalar", _sha256(ordered)).name(f"version_{role}"))


def _combine(tables: list[ir.Table]) -> ir.Table:
    first, *rest = tables
    return first.cross_join(*rest) if rest else first


def _generation(source_ids: Mapping[FeedRole, str], versions: Mapping[FeedRole, str]) -> str:
    return "duckdb-generation-v1:" + _identity(
        "increment.overlap.duckdb.generation.v1",
        [(role, source_ids[role], versions[role]) for role in FEED_ROLES],
    )


def _versions(row: Mapping[str, Any]) -> dict[FeedRole, str]:
    return {role: f"duckdb-rows-v1:{row[f'version_{role}']}" for role in FEED_ROLES}


def _query(cursor: DuckDBPyConnection, backend: Backend, expr: ir.Table) -> list[dict[str, Any]]:
    slots: dict[str, str] = {}

    def bind_text(node: exp.Expr) -> exp.Expr:
        if not isinstance(node, exp.Literal) or not node.is_string:
            return node
        value = cast("str", node.this)
        name = slots.setdefault(value, f"v{len(slots)}")
        return exp.Cast(this=exp.Placeholder(this=name), to=exp.DataType.build("VARCHAR"))

    # Ibis reuses AST nodes, so transform the copy to replace every occurrence.
    query = backend.compiler.to_sqlglot(expr).transform(bind_text)
    result = cursor.execute(
        query.sql(dialect=backend.dialect, copy=False, unsupported_level=sqlglot.ErrorLevel.RAISE),
        {name: value for value, name in slots.items()},
    )
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def _sealed_duckdb_generation(con: Backend) -> tuple[str, str] | None:
    """Identify a checkpointed DuckDB input opened as the only read-only catalog."""
    if con.name != "duckdb":
        return None
    rows = con.con.execute(
        "SELECT database_name, path, readonly FROM system.main.duckdb_databases() WHERE NOT internal"
    ).fetchall()
    if len(rows) != 1:
        return None
    catalog, location, read_only = rows[0]
    if not read_only or location is None:
        return None
    try:
        path = Path(location).resolve(strict=True)
        if not path.is_file() or path.with_suffix(path.suffix + ".wal").exists():
            return None
        with path.open("rb") as database:
            return catalog, file_digest(database, "sha256").hexdigest()
    except OSError:
        return None


def _interaction_coordinate(table: ir.Table) -> tuple[str, str, str] | None:
    op = table.op()
    if isinstance(op, ops.DatabaseTable) and op.namespace.catalog and op.namespace.database:
        return op.namespace.catalog, op.namespace.database, op.name
    return None


def _native_interaction_fact(backend: Backend, table: ir.Table) -> SourceFact | None:
    coordinate = _interaction_coordinate(table)
    op = table.op()
    if coordinate is None or not isinstance(op, ops.DatabaseTable) or op.source is not backend:
        return None
    rows = backend.con.execute(
        _CATALOG,
        [list(table.columns), *(part.translate(_ASCII_FOLD) for part in coordinate)],
    ).fetchall()
    if not rows or any(row[7] for row in rows) or not _uncollated_table(rows[0][4]):
        return None
    _, catalog, schema, name, *_ = rows[0]
    ordered = table.select(*sorted(table.columns))
    layout = tuple((key, str(dtype)) for key, dtype in ordered.schema().items())
    with backend.con.cursor() as cursor:
        cursor.execute("SET TimeZone = 'UTC'")
        cursor.execute("SET default_collation = ''")
        (digest,) = _query(cursor, backend, _feed_digest(ordered, "roster"))
    return SourceFact(
        coordinate=(catalog, schema, name),
        content_sha256=_identity(
            "increment.interaction.fact.v1", (layout, digest["version_roster"])
        ),
    )


def interaction_duckdb_fact(table: ir.Table) -> SourceFact:
    """Fingerprint one qualified native DuckDB table for a published source manifest."""
    op = table.op()
    if isinstance(op, ops.DatabaseTable) and isinstance(op.source, Backend):
        try:
            fact = _native_interaction_fact(op.source, table)
        except (DuckDBError, IbisError, OSError, SqlglotError):
            fact = None
        if fact is not None:
            return fact
    refuse(SCAN_INVALID, field="source", reason="native_facts_required")


def _published_generation_id(backend: Backend, catalog: str) -> str | None:
    try:
        marker = backend.table("increment_interaction_generation", database=(catalog, "main"))
        if _native_interaction_fact(backend, marker) is None:
            return None
        records = _query(backend.con, backend, marker.select("version", "generation_id").limit(2))
    except (DuckDBError, IbisError, OSError, KeyError):
        return None
    if len(records) != 1 or records[0]["version"] != 1:
        return None
    identifier = records[0]["generation_id"]
    return identifier if isinstance(identifier, str) and identifier else None


def _sealed_focal_fact(
    backend: Backend, catalog: str, query: str, pinned: ir.Table, original: ir.Table
) -> SourceFact | None:
    """Check the actual pinned projection against its source-owned native input."""
    try:
        select = sqlglot.parse_one(query, dialect="duckdb")
        if (
            not isinstance(select, exp.Select)
            or select.expressions != [exp.Star()]
            or any(
                value for key, value in select.args.items() if key not in {"expressions", "from_"}
            )
        ):
            return None
        source = select.args["from_"].this
        if (
            not isinstance(source, exp.Table)
            or not isinstance(source.this, exp.Identifier)
            or not source.db
            or any(
                value for key, value in source.args.items() if key not in {"this", "db", "catalog"}
            )
        ):
            return None
        if source.catalog and source.catalog.translate(_ASCII_FOLD) != catalog.translate(
            _ASCII_FOLD
        ):
            return None
        physical = backend.table(source.name, database=(catalog, source.db))
        expected = _feed_digest(original, "roster")
        observed = _feed_digest(pinned, "outcomes")
        (digests,) = _query(backend.con, backend, expected.cross_join(observed))
        if digests["version_roster"] != digests["version_outcomes"]:
            return None
        return _native_interaction_fact(backend, physical)
    except (DuckDBError, IbisError, OSError, SqlglotError, KeyError):
        return None


def _focal_query(source: Any, fact: str | None) -> str | None:
    for relation in source._defs.fact_sources:
        if any(item.name == fact for item in relation.facts):
            return relation.sql if not relation.dims else None
    return None


def interaction_focal_facts(source: Any, plan: InteractionScanPlan) -> FocalSourceFacts | None:
    """Read only the declared projections already pinned for the definitions reader."""
    session = source._session
    backend = session.con
    projections = session._source_tables
    originals = session._snapshot_inputs
    if not isinstance(backend, Backend) or projections is None or originals is None:
        return None
    generation = _sealed_duckdb_generation(backend)
    snapshot = session.source_snapshot_evidence
    if generation is None or snapshot is None:
        return None
    catalog, dataset = generation
    identifier = _published_generation_id(backend, catalog)
    if identifier is None:
        return None
    plan.scope.require_cutoff(snapshot.observation_cutoff_ts)
    definition = source._experiment
    ledger = definition.assignment_ledger
    exposure = source._exp_lookup_exposures[definition.exposure]
    exposure_sql = exposure.sql or _focal_query(source, exposure.fact)
    assignment_sql = ledger.sql if ledger is not None else exposure_sql
    by_query: dict[str, SourceFact] = {}

    def verify(query: str | None) -> SourceFact | None:
        if query is None or query not in projections or query not in originals:
            return None
        if query not in by_query:
            fact = _sealed_focal_fact(backend, catalog, query, projections[query], originals[query])
            if fact is None:
                return None
            by_query[query] = fact
        return by_query[query]

    assignments, exposures = verify(assignment_sql), verify(exposure_sql)
    if assignments is None or exposures is None:
        return None
    metrics = {metric.name: metric for metric in source._defs.metrics}
    feed_names = {fact.name: feed.name for feed in source._defs.fact_sources for fact in feed.facts}
    outcomes = []
    horizons = []
    for planned in plan.metrics:
        metric = metrics[planned.name]
        fact = verify(_focal_query(source, metric.fact))
        if fact is None:
            return None
        edge, certified = source._metric_snapshot_edges(metric, snapshot)
        if edge < plan.scope.end.date() - timedelta(days=1):
            refuse(SCAN_INVALID, field="source", reason="focal_outcome_window_incomplete")
        watermark = snapshot.complete_through_by_feed.get(feed_names[metric.fact])
        outcomes.append((planned.name, fact))
        horizons.append((planned.name, watermark, edge, certified))
    return FocalSourceFacts(
        generation_id=identifier,
        dataset_sha256=dataset,
        assignments=assignments,
        exposures=exposures,
        outcomes=tuple(sorted(outcomes)),
        horizons=tuple(sorted(horizons)),
    )


def interaction_pair_facts(
    source: Any,
    evidence: OverlapEvidence,
    tables: Mapping[str, tuple[str, str, str]],
) -> tuple[StudyFeedFact, ...]:
    """Check bounded warehouse evidence against the original native relations."""
    backend = source._session.con
    if evidence.ingress != "warehouse" or frozenset(tables) != frozenset(FEED_ROLES):
        refuse(SCAN_INVALID, field="pair_sources", reason="warehouse_receipt_required")
    try:
        roster, decisions, deliveries, outcomes = (
            backend.table(name, database=(catalog, schema))
            for role in FEED_ROLES
            for catalog, schema, name in (tables[role],)
        )
    except (DuckDBError, IbisError, OSError):
        refuse(SCAN_INVALID, field="pair_sources", reason="pair_relation_unavailable")
    verified = capture_overlap_warehouse(
        evidence.plan,
        capture=evidence.capture,
        roster=roster,
        decisions=decisions,
        deliveries=deliveries,
        outcomes=outcomes,
    )
    if verified.evidence_sha256 != evidence.evidence_sha256:
        refuse(SCAN_INVALID, field="pair_sources", reason="pair_evidence_mismatch")
    return tuple(
        StudyFeedFact(
            study_id=evidence.plan.study_id,
            role=role,
            fact=SourceFact(
                coordinate=tables[role],
                content_sha256=evidence.capture.feed(role).version.removeprefix("duckdb-rows-v1:"),
            ),
        )
        for role in FEED_ROLES
    )


def bind_overlap_warehouse(
    plan: OverlapPlan,
    *,
    capture: OverlapCapture,
    roster: ir.Table,
    decisions: ir.Table,
    deliveries: ir.Table,
    outcomes: ir.Table,
) -> OverlapCapture:
    """Bind producer coverage to the named DuckDB tables and their current contents.

    The source IDs, versions, and generation on ``capture`` are replaced. The
    producer still certifies timestamps, coverage, and the archived assignment law.
    """
    _validate_declarations(plan, capture)
    sources = _sources(
        plan,
        {"roster": roster, "decisions": decisions, "deliveries": deliveries, "outcomes": outcomes},
    )
    try:
        with _snapshot(sources) as cursor:
            source_ids = _locators(cursor, sources)
            digests = _combine(
                [_feed_digest(table, role) for role, table in sources.scoped(plan).items()]
            )
            versions = _versions(_query(cursor, sources.backend, digests)[0])
        feeds = tuple(
            capture.feed(role).model_copy(
                update={"source_id": source_ids[role], "version": versions[role]}
            )
            for role in FEED_ROLES
        )
        return capture.model_copy(
            update={"generation_id": _generation(source_ids, versions), "feeds": feeds}
        )
    except (DuckDBError, IbisError, OSError, SqlglotError):
        pass
    _unbound()


@dataclass
class _Checks:
    tables: list[ir.Table] = field(default_factory=list)
    labels: list[tuple[str, str, FactorId | None]] = field(default_factory=list)

    def add(self, table: ir.Table, role: str, **predicates: ir.BooleanValue) -> None:
        counts = []
        for reason, predicate in predicates.items():
            counts.append(table.count(where=predicate).name(f"violation_{len(self.labels)}"))
            self.labels.append((role, reason, None))
        self.tables.append(table.aggregate(counts))

    def count(
        self, table: ir.Table, role: str, reason: str, factor: FactorId | None = None
    ) -> None:
        self.tables.append(table.aggregate(table.count().name(f"violation_{len(self.labels)}")))
        self.labels.append((role, reason, factor))

    def summary(self) -> ir.Table:
        return _combine(self.tables)

    def check(self, row: Mapping[str, Any]) -> None:
        for i, (role, reason, factor) in enumerate(self.labels):
            n = row[f"violation_{i}"]
            if n and factor is not None:
                refuse_assignment_invalid(reason, details=(factor, n))
            if n:
                _evidence_error(role, reason, n)


def _zero_violations(summary: ir.Table) -> ir.BooleanValue:
    return cast(
        "ir.BooleanValue",
        ibis.and_(
            *(summary[name] == 0 for name in summary.columns if name.startswith("violation_"))
        ),
    )


def _guard(table: ir.Table, condition: ir.Table) -> ir.Table:
    return table.cross_join(condition).select(table)


def _scalars(table: ir.Table, columns: tuple[OverlapColumn, ...]) -> ir.BooleanValue:
    invalid = []
    for column in columns:
        value = table[column.name]
        if not column.nullable:
            invalid.append(value.isnull())
        if column.kind == "string":
            invalid.append(value == "")
        if column.kind == "timestamp":
            finite = cast("ir.BooleanValue", _finite_timestamp(value))
            invalid.append(value.notnull() & ~finite)
    return cast("ir.BooleanValue", ibis.or_(*invalid))


def _checked_keys(
    plan: OverlapPlan, sources: _Sources, scoped: Mapping[FeedRole, ir.Table], checks: _Checks
) -> dict[FeedRole, ir.Table]:
    tables = dict(scoped)
    a, b = plan.a, plan.b
    decisions, deliveries, outcomes = (
        scoped[role] for role in ("decisions", "deliveries", "outcomes")
    )
    tables["decisions"] = decisions.filter(
        decisions.target.isnull() | decisions.target.isin(("", a.target, b.target))
    )
    tables["deliveries"] = deliveries.filter(
        deliveries.experiment.isnull()
        | deliveries.experiment.isin(("", a.experiment, b.experiment))
    )
    tables["outcomes"] = outcomes.filter(
        outcomes.metric.isnull() | outcomes.metric.isin(["", *[m.name for m in plan.metrics]])
    )
    for role, table in tables.items():
        checks.add(table, role, invalid_value=_scalars(table, sources.columns[role]))
    for role, table in tables.items():
        key_counts = table.group_by(*OVERLAP_RELATIONS[role].key).aggregate(n=table.count())
        checks.count(key_counts.filter(key_counts.n > 1), role, "duplicate_key")
    key_summary = checks.summary()
    guard = key_summary.filter(_zero_violations(key_summary))
    return {role: _guard(table, guard) for role, table in tables.items()}


def _decisions_spine(roster: ir.Table, decisions: ir.Table, plan: OverlapPlan) -> ir.Table:
    def attempt(factor: OverlapFactor, prefix: str) -> ir.Table:
        scoped = decisions.filter(decisions.target == factor.target)
        return scoped.select(
            *_KEYS,
            **{f"{prefix}_{name}": scoped[name] for name in scoped.columns if name not in _KEYS},
        )

    a = attempt(plan.a, "a")
    b = attempt(plan.b, "b")
    joined = roster.left_join(a, list(_KEYS)).select(
        roster, *[a[name] for name in a.columns if name not in _KEYS]
    )
    return joined.left_join(b, list(_KEYS)).select(
        joined, *[b[name] for name in b.columns if name not in _KEYS]
    )


def _check_entry(spine: ir.Table, plan: OverlapPlan, checks: _Checks) -> None:
    for reason, expect in (("unexpected_decision", False), ("missing_decision", True)):
        for factor in FACTOR_IDS:
            key = factor.lower()
            present = spine[f"{key}_target"].notnull()
            eligible = spine[f"eligible_{key}"]
            invalid = eligible & ~present if expect else ~eligible & present
            checks.count(spine.filter(invalid), "decisions", reason, factor)
    last_eligibility = cast(
        "ir.TimestampValue", ibis.greatest(spine.eligibility_at_a, spine.eligibility_at_b)
    )
    checks.add(
        spine,
        "roster",
        entry_outside_window=(spine.entry_at < plan.entry_open) | (spine.entry_at >= plan.start),
        eligibility_before_entry=(spine.eligibility_at_a < spine.entry_at)
        | (spine.eligibility_at_b < spine.entry_at),
        eligibility_after_start=(spine.eligibility_at_a > plan.start)
        | (spine.eligibility_at_b > plan.start),
        eligibility_not_before_either_decision=(spine.a_decided_at <= last_eligibility)
        | (spine.b_decided_at <= last_eligibility),
    )


def _factor_values(decisions: ir.Table, plan: OverlapPlan, attr: str) -> ir.Value:
    return (decisions.target == plan.a.target).ifelse(getattr(plan.a, attr), getattr(plan.b, attr))


def _check_decisions(decisions: ir.Table, plan: OverlapPlan, checks: _Checks) -> None:
    expected_archive = (decisions.target == plan.a.target).ifelse(*plan.law.archive_sha256)
    assigned = decisions.status == "assigned"
    not_enrolled = decisions.status == "not_enrolled"
    identity = decisions.selected_experiment.notnull()
    arm = decisions.arm.notnull()
    weight = decisions.assigned_weight.notnull()
    focal = decisions.selected_experiment == _factor_values(decisions, plan, "experiment")
    off = (decisions.arm == _factor_values(decisions, plan, "off")) & (
        decisions.assigned_weight == _factor_values(decisions, plan, "off_weight")
    )
    on = (decisions.arm == _factor_values(decisions, plan, "on")) & (
        decisions.assigned_weight == _factor_values(decisions, plan, "on_weight")
    )
    standalone = [
        factor.target
        for factor in (plan.a, plan.b)
        if plan.law.origin == "sdk_v1" and factor.layer is None
    ]
    layer_only = decisions.target.isin(standalone) & (
        (decisions.status == "unallocated") | (identity & ~focal)
    )
    checks.add(
        decisions,
        "decisions",
        revision_mismatch=decisions.revision != _factor_values(decisions, plan, "revision"),
        archive_mismatch=decisions.archive_sha256 != expected_archive,
        decision_after_start=decisions.decided_at > plan.start,
        invalid_status=~decisions.status.isin(DECISION_STATUSES),
        invalid_disposition=(
            assigned
            & (
                ~identity
                | ~arm
                | ~weight
                | (decisions.assigned_weight <= 0)
                | (decisions.assigned_weight > MAX_EXACT_INTEGER)
                | (focal & ~(off | on))
            )
        )
        | (not_enrolled & (~identity | arm | weight))
        | (~assigned & ~not_enrolled & (identity | arm | weight))
        | layer_only,
    )


def _with_gates(spine: ir.Table, plan: OverlapPlan) -> ir.Table:
    def gate(prefix: str, factor: OverlapFactor) -> ir.Value:
        other = spine[f"{prefix}_selected_experiment"] != factor.experiment
        return other.fill_null(False).ifelse("other_selected", spine[f"{prefix}_status"])

    return spine.mutate(a_disposition=gate("a", plan.a), b_disposition=gate("b", plan.b))


def _check_gates(spine: ir.Table, plan: OverlapPlan, checks: _Checks) -> None:
    both = spine.eligible_a & spine.eligible_b
    assigned = (spine.a_disposition == "assigned") & (spine.b_disposition == "assigned")
    if plan.law.gate_basis == "preassigned_roster":
        latest = ibis.greatest(spine.gate_at_a, spine.gate_at_b)
        checks.add(
            spine,
            "roster",
            missing_gate=both
            & (
                spine.gate_a.isnull()
                | spine.gate_b.isnull()
                | spine.gate_at_a.isnull()
                | spine.gate_at_b.isnull()
            ),
            gate_out_of_order=both
            & (
                (spine.gate_at_a < spine.eligibility_at_a)
                | (spine.gate_at_b < spine.eligibility_at_b)
                | (latest >= spine.a_decided_at)
                | (latest >= spine.b_decided_at)
            ),
            gate_mismatch=both
            & (
                (spine.gate_a != (spine.a_disposition == "assigned"))
                | (spine.gate_b != (spine.b_disposition == "assigned"))
            ),
        )
    if plan.law.admission == "both_guaranteed":
        checks.add(spine, "decisions", admission_count_mismatch=both & ~assigned)


def _check_outcomes(
    outcomes: ir.Table, plan: OverlapPlan, capture: OverlapCapture, checks: _Checks
) -> None:
    expected = ibis.cases(*[(outcomes.metric == m.name, m.definition_sha256) for m in plan.metrics])
    binary = outcomes.metric.isin([m.name for m in plan.metrics if m.kind == "binary_mean"])
    checks.add(
        outcomes,
        "outcomes",
        window_mismatch=(outcomes.window_start != plan.start) | (outcomes.window_end != plan.end),
        definition_mismatch=outcomes.definition_sha256 != expected,
        finalized_after_cutoff=outcomes.finalized_at > capture.cutoff,
        finalized_before_end=outcomes.finalized_at < plan.end,
        invalid_value=outcomes.value.isnan()
        | outcomes.value.isinf()
        | (binary & ~outcomes.value.isin((0.0, 1.0))),
    )


def _tallies(spine: ir.Table) -> ir.Table:
    def bit(flag: ir.BooleanValue, value: str) -> ir.BooleanValue:
        return flag if value == "1" else ~flag

    eligibility = {
        f"eligibility_{cell}": spine.count(
            where=bit(spine.eligible_a, cell[0]) & bit(spine.eligible_b, cell[1])
        )
        for cell in CELL_IDS
    }
    both = spine.eligible_a & spine.eligible_b
    gates = {
        f"gate_{index}": spine.count(
            where=both & (spine.a_disposition == a) & (spine.b_disposition == b)
        )
        for index, (a, b) in enumerate(GATE_KEYS)
    }
    return spine.aggregate([count.name(name) for name, count in (eligibility | gates).items()])


def _joint_spine(spine: ir.Table, plan: OverlapPlan) -> ir.Table:
    target = spine.filter(
        spine.eligible_a & spine.eligible_b,
        spine.a_disposition == "assigned",
        spine.b_disposition == "assigned",
    )
    cell = (target.a_arm == plan.a.on).ifelse("1", "0") + (target.b_arm == plan.b.on).ifelse(
        "1", "0"
    )
    return target.select(*_KEYS, cell=cell)


def _delivery_counts(
    deliveries: ir.Table, spine: ir.Table, target: ir.Table, plan: OverlapPlan
) -> ir.Table:
    events = deliveries.inner_join(spine, list(_KEYS)).select(
        deliveries, spine.a_disposition, spine.b_disposition, spine.a_arm, spine.b_arm
    )

    def agrees(prefix: str, factor: OverlapFactor) -> ir.BooleanValue:
        return (
            (events.target == factor.target)
            & (events.revision == factor.revision)
            & (events[f"{prefix}_disposition"] == "assigned")
            & (events.delivered_arm == events[f"{prefix}_arm"])
        ).fill_null(False)

    a_event = events.experiment == plan.a.experiment
    events = events.mutate(
        factor=a_event.ifelse("A", "B"),
        matches=a_event.ifelse(agrees("a", plan.a), agrees("b", plan.b)),
    )
    unit_counts = events.group_by(*_KEYS, "factor").aggregate(
        n_matching=events.count(where=events.matches),
        n_inconsistent=events.count(where=~events.matches),
    )
    opportunities = target.mutate(factor=ibis.literal("A")).union(
        target.mutate(factor=ibis.literal("B"))
    )
    joined = opportunities.left_join(unit_counts, [*_KEYS, "factor"]).select(
        opportunities, unit_counts.n_matching, unit_counts.n_inconsistent
    )
    inside = joined.group_by("cell", "factor").aggregate(
        n_without_delivery=joined.n_matching.isnull().sum(),
        n_matching=joined.n_matching.sum().fill_null(0),
        n_inconsistent=joined.n_inconsistent.sum().fill_null(0),
    )
    outside_units = unit_counts.anti_join(target, list(_KEYS))
    outside = (
        outside_units.group_by("factor")
        .aggregate(n_inconsistent=outside_units.n_inconsistent.sum())
        .select(
            cell=ibis.literal("outside_target"),
            factor="factor",
            n_without_delivery=ibis.literal(0, type="int64"),
            n_matching=ibis.literal(0, type="int64"),
            n_inconsistent="n_inconsistent",
        )
    )
    return inside.union(outside)


def _metric_observations(
    target: ir.Table, outcomes: ir.Table, plan: OverlapPlan, capture: OverlapCapture
) -> ir.Table:
    names = cast("ir.ArrayScalar", ibis.literal([metric.name for metric in plan.metrics]))
    metrics = names.unnest().name("metric").as_table()
    skeleton = target.cross_join(metrics)
    matched = skeleton.left_join(outcomes, [*_KEYS, "metric"]).select(
        skeleton.cell,
        skeleton.metric,
        outcomes.value,
        outcomes.finalized_at,
        present=outcomes.unit_id.notnull(),
    )
    covered = capture.outcomes_cover(plan)
    pending = (
        matched.finalized_at.isnull() & matched.present
        if covered
        else cast("ir.BooleanValue", ibis.literal(True))
    )
    observed = ~pending & matched.value.notnull()
    mean = matched.metric.isin([metric.name for metric in plan.metrics if metric.kind == "mean"])
    return matched.select(
        matched.metric,
        matched.cell,
        pending=pending,
        observed=observed,
        success=observed & (matched.value == 1.0),
        mean_value=(mean & observed).ifelse(matched.value.cast("float64"), ibis.null("float64")),
    )


def _cell_moments(observations: ir.Table) -> ir.Table:
    keys = ["metric", "cell"]
    counts = observations.group_by(*keys).aggregate(
        n_assigned=observations.count(),
        n_observed=observations.count(where=observations.observed),
        n_pending=observations.count(where=observations.pending),
        n_success=observations.count(where=observations.success),
        anchor=observations.mean_value.min(),
        maximum=observations.mean_value.max(),
    )
    complete = counts.filter(counts.n_assigned == counts.n_observed, counts.anchor.notnull())
    anchors = complete.select(*keys, complete.anchor, spread=complete.maximum - complete.anchor)
    residuals = observations.inner_join(anchors, keys).select(
        *[observations[key] for key in keys],
        anchors.spread,
        offset=observations.mean_value - anchors.anchor,
    )
    centers = residuals.group_by(*keys).aggregate(mean_delta=_compensated_mean(residuals.offset))
    deviations = residuals.inner_join(centers, keys).select(
        *[residuals[key] for key in keys],
        residuals.spread,
        # Scaling before squaring avoids losing individually subnormal terms.
        scaled=(residuals.spread == 0).ifelse(
            0.0, (residuals.offset - centers.mean_delta) / residuals.spread
        ),
    )
    norm = deviations.group_by(*keys).aggregate(
        qsum=_compensated_sum(deviations.scaled * deviations.scaled),
        spread=deviations.spread.max(),
    )
    ss = norm.select(*keys, norm.spread, m2=(norm.qsum * norm.spread) * norm.spread)
    stats = centers.inner_join(ss, keys).select(centers, ss.spread, ss.m2)
    return counts.left_join(stats, keys).select(
        *[counts[name] for name in counts.columns if name not in ("anchor", "maximum")],
        stats.spread,
        stats.mean_delta,
        stats.m2,
        anchor=(counts.n_observed == counts.n_assigned).ifelse(counts.anchor, ibis.null("float64")),
    )


def _union_sections(sections: tuple[ir.Table, ...]) -> ir.Table:
    # Only the summary carries integrity counts and digests, and each cell is emitted once.
    schema = {name: dtype for section in sections for name, dtype in section.schema().items()}
    combined = [
        section.select(
            section=ibis.literal(index),
            **{
                name: section[name] if name in section.columns else ibis.null(dtype)
                for name, dtype in schema.items()
            },
        )
        for index, section in enumerate(sections)
    ]
    return combined[0].union(*combined[1:])


def _aggregate_capture(
    plan: OverlapPlan,
    capture: OverlapCapture,
    sources: _Sources,
    scoped: Mapping[FeedRole, ir.Table],
    digests: ir.Table,
) -> tuple[ir.Table, _Checks]:
    checks = _Checks()
    tables = _checked_keys(plan, sources, scoped, checks)
    roster, decisions, deliveries, outcomes = (tables[role] for role in FEED_ROLES)
    for role, table in (("decisions", decisions), ("deliveries", deliveries)):
        checks.count(table.anti_join(roster, list(_KEYS)), role, "orphan_unit")
    spine = _decisions_spine(roster, decisions, plan)
    _check_entry(spine, plan, checks)
    _check_decisions(decisions, plan, checks)
    spine = _with_gates(spine, plan)
    _check_gates(spine, plan, checks)
    events = deliveries.inner_join(roster, list(_KEYS)).select(
        deliveries.delivered_at, roster.eligibility_at_a, roster.eligibility_at_b
    )
    checks.add(
        events,
        "deliveries",
        delivery_before_start=events.delivered_at < plan.start,
        delivery_after_cutoff=events.delivered_at > capture.cutoff,
        eligibility_not_before_delivery=(events.eligibility_at_a >= events.delivered_at)
        | (events.eligibility_at_b >= events.delivered_at),
    )
    _check_outcomes(outcomes, plan, capture, checks)
    proof = checks.summary().cross_join(digests)
    verified = proof.filter(
        _zero_violations(proof),
        *[
            ("duckdb-rows-v1:" + proof[f"version_{role}"]) == capture.feed(role).version
            for role in FEED_ROLES
        ],
    )
    target = _guard(_joint_spine(spine, plan), verified)
    cells = _cell_moments(_metric_observations(target, outcomes, plan, capture))
    in_window = deliveries.filter(deliveries.delivered_at < plan.end)
    delivery_counts = _delivery_counts(_guard(in_window, verified), spine, target, plan)
    return _union_sections((proof.cross_join(_tallies(spine)), delivery_counts, cells)), checks


def _cell_values(metric: OverlapMetric, row: Mapping[str, Any]) -> dict[str, Any]:
    assigned = row.get("n_assigned", 0)
    observed = row.get("n_observed", 0)
    pending = row.get("n_pending", 0)
    binary = metric.kind == "binary_mean"
    success = row.get("n_success", 0) if binary else None
    anchor = delta = m2 = None
    if observed and assigned == observed:
        if success is not None:
            anchor, delta, m2 = overlap_binary_moments(observed, success)
        else:
            anchor, delta, m2 = row["anchor"], row["mean_delta"], row["m2"]
            spread = row["spread"]
            if not isfinite(spread):
                refuse(NUMERIC_RANGE, metric=metric.name, quantity="centered_range")
            if not isfinite(anchor + delta) or (spread > 0 and delta == 0):
                refuse(NUMERIC_RANGE, metric=metric.name, quantity="mean_delta")
            if not isfinite(m2) or (spread > 0 and m2 == 0):
                refuse(NUMERIC_RANGE, metric=metric.name, quantity="m2")
            if anchor == 0.0:
                anchor = 0.0
    return {
        "metric": metric.name,
        "n_assigned": assigned,
        "n_observed": observed,
        "n_missing": assigned - observed - pending,
        "n_pending": pending,
        "n_success": success,
        "anchor": anchor,
        "mean_delta": delta,
        "m2": m2,
    }


def _make_cell(metric: OverlapMetric, cell: CellId, row: Mapping[str, Any]) -> OverlapCell:
    return OverlapCell(cell=cell, **_cell_values(metric, row))


def _evidence(
    plan: OverlapPlan, capture: OverlapCapture, rows: list[dict[str, Any]], checks: _Checks
) -> OverlapEvidence:
    (summary,) = (row for row in rows if row["section"] == 0)
    if any(capture.feed(role).version != version for role, version in _versions(summary).items()):
        _unbound()
    checks.check(summary)
    deliveries = {(row["cell"], row["factor"]): row for row in rows if row["section"] == 1}
    outcomes = {(row["metric"], row["cell"]): row for row in rows if row["section"] == 2}
    return OverlapEvidence(
        plan=plan,
        capture=capture,
        ingress="warehouse",
        eligibility_counts=tuple(summary[f"eligibility_{cell}"] for cell in CELL_IDS),
        gate_counts=tuple(summary[f"gate_{index}"] for index in range(len(GATE_KEYS))),
        delivery_counts=tuple(
            OverlapDeliveryCount(
                cell=cell,
                factor=factor,
                n_without_delivery=row.get("n_without_delivery", 0),
                n_matching=row.get("n_matching", 0),
                n_inconsistent=row.get("n_inconsistent", 0),
            )
            for cell, factor in DELIVERY_KEYS
            for row in (deliveries.get((cell, factor), {}),)
        ),
        cells=tuple(
            _make_cell(metric, cell, outcomes.get((metric.name, cell), {}))
            for metric in plan.metrics
            for cell in CELL_IDS
        ),
    )


def capture_overlap_warehouse(
    plan: OverlapPlan,
    *,
    capture: OverlapCapture,
    roster: ir.Table,
    decisions: ir.Table,
    deliveries: ir.Table,
    outcomes: ir.Table,
) -> OverlapEvidence:
    """Capture bounded counts and moments from a verified DuckDB generation.

    Use catalog- and schema-qualified local native tables on one connection.
    Obtain ``capture`` from :func:`bind_overlap_warehouse`. The capture checks
    the source versions and computes all counts and moments in one statement;
    it never transfers source rows into Python.
    """
    _validate_declarations(plan, capture)
    sources = _sources(
        plan,
        {"roster": roster, "decisions": decisions, "deliveries": deliveries, "outcomes": outcomes},
    )
    versions = {role: capture.feed(role).version for role in FEED_ROLES}
    try:
        with _snapshot(sources) as cursor:
            source_ids = _locators(cursor, sources)
            if any(capture.feed(role).source_id != source_ids[role] for role in FEED_ROLES):
                _unbound()
            if capture.generation_id != _generation(source_ids, versions):
                _unbound()
            scoped = sources.scoped(plan)
            digests = _combine([_feed_digest(table, role) for role, table in scoped.items()])
            expr, checks = _aggregate_capture(plan, capture, sources, scoped, digests)
            rows = _query(cursor, sources.backend, expr)
        return _evidence(plan, capture, rows, checks)
    except (DuckDBError, IbisError, OSError, SqlglotError):
        pass
    # A driver error can contain source values, SQL literals or filesystem paths.
    _unbound()


_JOINT_ROSTER = (OverlapColumn("unit_id", "string"), OverlapColumn("assignment", "string"))
_JOINT_OUTCOMES = (
    OverlapColumn("unit_id", "string"),
    OverlapColumn("metric", "string"),
    OverlapColumn("definition_sha256", "string"),
    OverlapColumn("value", "float", nullable=True),
    OverlapColumn("finalized_at", "timestamp", nullable=True),
)


def _joint_relation(
    backend: Backend, catalog: str, fact: SourceFact, columns: tuple[OverlapColumn, ...], role: str
) -> ir.Table:
    source_catalog, schema, name = fact.coordinate
    if source_catalog != catalog:
        refuse(SCAN_INVALID, field="combination_feeds", reason="different_source_generation")
    table = backend.table(name, database=(source_catalog, schema))
    if _native_interaction_fact(backend, table) != fact:
        refuse(SCAN_INVALID, field="combination_feeds", reason="combination_evidence_mismatch")
    types = table.schema()
    if any(col.name not in types or not _valid_type(col.kind, types[col.name]) for col in columns):
        _evidence_error(role, "invalid_column_type", 1)
    return table.select(*(col.name for col in columns))


def _joint_moments(
    roster: ir.Table,
    outcomes: ir.Table,
    plan: InteractionScanPlan,
    combination: InteractionCombination,
) -> tuple[ir.Table, _Checks]:
    checks = _Checks()
    binary_metrics = [m.name for m in plan.metrics if m.kind == "binary_mean"]
    checks.add(
        roster,
        "roster",
        invalid_value=_scalars(roster, _JOINT_ROSTER),
        invalid_assignment=~roster.assignment.isin(combination.assignments),
    )
    checks.add(
        outcomes,
        "outcomes",
        invalid_value=_scalars(outcomes, _JOINT_OUTCOMES)
        | outcomes.value.isnan()
        | outcomes.value.isinf()
        | (outcomes.metric.isin(binary_metrics) & ~outcomes.value.isin((0.0, 1.0))),
        unknown_metric=~outcomes.metric.isin([m.name for m in plan.metrics]),
        definition_mismatch=outcomes.definition_sha256
        != ibis.cases(*((outcomes.metric == m.name, m.definition_sha256) for m in plan.metrics)),
        finalized_after_cutoff=outcomes.finalized_at > plan.scope.cutoff,
        finalized_before_end=outcomes.finalized_at < plan.scope.end,
    )
    for role, table, keys in (
        ("roster", roster, ("unit_id",)),
        ("outcomes", outcomes, ("unit_id", "metric")),
    ):
        distinct = table.group_by(*keys).aggregate(n=table.count())
        checks.count(distinct.filter(distinct.n > 1), role, "duplicate_key")
    checks.count(outcomes.anti_join(roster, "unit_id"), "outcomes", "orphan_unit")
    proof = checks.summary()
    names = cast("ir.ArrayScalar", ibis.literal([m.name for m in plan.metrics]))
    skeleton = _guard(roster, proof.filter(_zero_violations(proof))).cross_join(
        names.unnest().name("metric").as_table()
    )
    joined = skeleton.left_join(outcomes, ["unit_id", "metric"]).select(
        cell=skeleton.assignment,
        metric=skeleton.metric,
        value=outcomes.value,
        observed=outcomes.finalized_at.notnull() & outcomes.value.notnull(),
        pending=outcomes.unit_id.notnull() & outcomes.finalized_at.isnull(),
    )
    stats = _cell_moments(
        joined.select(
            joined.cell,
            joined.metric,
            joined.pending,
            joined.observed,
            success=joined.observed & (joined.value == 1.0),
            mean_value=(~joined.metric.isin(binary_metrics) & joined.observed).ifelse(
                joined.value.cast("float64"), ibis.null("float64")
            ),
        )
    )
    return _union_sections((proof, stats)), checks


def interaction_combination_cells(
    source: Any,
    catalog: str,
    plan: InteractionScanPlan,
    combination: InteractionCombination,
    feed: CombinationFeed,
) -> tuple[CombinationCell, ...]:
    """Reduce native joint assignments rooted in the already qualified generation."""
    backend = source._session.con
    try:
        roster = _joint_relation(backend, catalog, feed.roster, _JOINT_ROSTER, "roster")
        outcomes = _joint_relation(backend, catalog, feed.outcomes, _JOINT_OUTCOMES, "outcomes")
        expr, checks = _joint_moments(roster, outcomes, plan, combination)
        with backend.con.cursor() as cursor:
            cursor.execute("SET TimeZone = 'UTC'")
            cursor.execute("SET default_collation = ''")
            records = _query(cursor, backend, expr)
    except (DuckDBError, IbisError, TableNotFound, OSError, SqlglotError, KeyError):
        refuse(SCAN_INVALID, field="combination_feeds", reason="joint_relation_unavailable")
    (proof,) = (row for row in records if row["section"] == 0)
    checks.check(proof)
    by_key = {(row["metric"], row["cell"]): row for row in records if row["section"] == 1}
    assignments = combination.assignments
    return tuple(
        CombinationCell(cell=bits, **_cell_values(metric, by_key.get((metric.name, bits), {})))
        for metric in plan.metrics
        for bits in assignments
    )


def interaction_combination_registered(
    source: Any, catalog: str, registration: CombinationRegistration
) -> bool:
    """Check the sealed publisher's pre-entry receipt, not the caller's timestamp."""
    backend = source._session.con
    try:
        table = backend.table(
            "increment_interaction_combination_registration", database=(catalog, "main")
        )
        types = table.schema()
        if (
            "registration_sha256" not in types
            or not types["registration_sha256"].is_string()
            or "registered_at" not in types
            or not _valid_type("timestamp", types["registered_at"])
            or _native_interaction_fact(backend, table) is None
        ):
            return False
        matching = table.filter(table.registration_sha256 == registration.registration_id)
        receipt = matching.aggregate(
            rows=matching.count(),
            exact=matching.count(where=matching.registered_at == registration.registered_at),
        )
        (record,) = _query(backend.con, backend, receipt)
        return record["rows"] == record["exact"] == 1
    except (DuckDBError, IbisError, TableNotFound, OSError, SqlglotError, KeyError):
        return False
