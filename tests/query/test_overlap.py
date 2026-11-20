"""DuckDB capture egress, revision integrity and source generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import ibis
import pytest
import sqlglot
from sqlglot import exp

from increment.errors import CodedError
from increment.query.overlap import bind_overlap_warehouse, capture_overlap_warehouse
from increment.semantics.overlap import OverlapCapture, OverlapPlan

pytestmark = pytest.mark.integration

_SCHEMAS = {
    "roster": {
        "study_id": "string",
        "unit_id": "string",
        "entry_at": "timestamp('UTC')",
        "eligible_a": "boolean",
        "eligible_b": "boolean",
        "eligibility_at_a": "timestamp('UTC')",
        "eligibility_at_b": "timestamp('UTC')",
        "gate_a": "boolean",
        "gate_b": "boolean",
        "gate_at_a": "timestamp('UTC')",
        "gate_at_b": "timestamp('UTC')",
    },
    "decisions": {
        "study_id": "string",
        "unit_id": "string",
        "target": "string",
        "decided_at": "timestamp('UTC')",
        "revision": "string",
        "archive_sha256": "string",
        "selected_experiment": "string",
        "status": "string",
        "arm": "string",
        "assigned_weight": "int64",
    },
    "deliveries": {
        "study_id": "string",
        "event_id": "string",
        "unit_id": "string",
        "target": "string",
        "experiment": "string",
        "revision": "string",
        "delivered_arm": "string",
        "delivered_at": "timestamp('UTC')",
    },
    "outcomes": {
        "study_id": "string",
        "unit_id": "string",
        "metric": "string",
        "definition_sha256": "string",
        "window_start": "timestamp('UTC')",
        "window_end": "timestamp('UTC')",
        "value": "float64",
        "finalized_at": "timestamp('UTC')",
    },
}


@pytest.fixture
def warehouse(tmp_path):
    path = tmp_path / "overlap.duckdb"
    backend = ibis.duckdb.connect(path)
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/overlap/contract_v1.json").read_text()
    )
    for role, fields in _SCHEMAS.items():
        rows = [
            {
                key: datetime.fromisoformat(value)
                if value is not None and fields[key].startswith("timestamp")
                else value
                for key, value in row.items()
            }
            for row in fixture["external"][role]
        ]
        schema = ibis.schema(fields)
        backend.create_table(
            role, ibis.memtable(rows, schema=schema) if rows else None, schema=schema
        )
    bindings = {role: backend.table(role, database=("overlap", "main")) for role in _SCHEMAS}
    plan = OverlapPlan.model_validate(fixture["evidence"]["plan"])
    template = OverlapCapture.model_validate(fixture["evidence"]["capture"])
    yield backend, path, plan, template, bindings, fixture["evidence"]
    backend.disconnect()


def test_common_entry_duckdb_fixture(warehouse):
    backend, _, plan, template, bindings, expected = warehouse
    backend.raw_sql(
        """INSERT INTO decisions SELECT * REPLACE ('unrelated-study' AS study_id,
            'wrong-revision' AS revision, 'off' AS arm) FROM decisions;
        INSERT INTO deliveries VALUES
            ('common-entry-v1', 'a1', '001', 'target-a', 'experiment-a', 'rev-a',
             'on', TIMESTAMPTZ '2026-01-01 00:00:04+00'),
            ('common-entry-v1', 'a2', '001', 'target-a', 'experiment-a', 'rev-a',
             'on', TIMESTAMPTZ '2026-01-01 00:00:04+00'),
            ('common-entry-v1', 'b1', '001', 'target-b', 'experiment-b', 'wrong-revision',
             'off', TIMESTAMPTZ '2026-01-01 00:00:04+00'),
            ('common-entry-v1', 'b2', 'x', 'target-b', 'experiment-b', 'rev-b',
             'off', TIMESTAMPTZ '2026-01-01 00:00:04+00')"""
    )
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    evidence = capture_overlap_warehouse(plan, capture=capture, **bindings)
    assert evidence.ingress == "warehouse"
    assert evidence.eligibility_counts == tuple(expected["eligibility_counts"])
    assert evidence.gate_counts == tuple(expected["gate_counts"])
    assert [cell.model_dump() for cell in evidence.cells] == expected["cells"]
    deliveries = [row.model_dump() for row in evidence.delivery_counts]
    expected["delivery_counts"][2].update(n_without_delivery=0, n_matching=2)
    expected["delivery_counts"][3].update(n_without_delivery=0, n_inconsistent=1)
    assert deliveries == expected["delivery_counts"]


def test_adjacent_floats_and_exact_binary_centers(warehouse):
    backend, _, plan, template, bindings, _ = warehouse
    backend.raw_sql(
        """INSERT INTO roster SELECT * REPLACE ('third' AS unit_id) FROM roster WHERE unit_id='001';
        INSERT INTO decisions SELECT * REPLACE ('third' AS unit_id) FROM decisions WHERE unit_id='001';
        UPDATE outcomes SET value=1e16 WHERE unit_id='001' AND metric='orders';
        INSERT INTO outcomes SELECT * REPLACE ('third' AS unit_id,
            CASE metric WHEN 'orders' THEN 10000000000000002::DOUBLE ELSE 1::DOUBLE END AS value)
            FROM outcomes WHERE unit_id='001'"""
    )
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    evidence = capture_overlap_warehouse(plan, capture=capture, **bindings)
    orders, binary = evidence.cells[1], evidence.cells[5]
    assert (orders.n_observed, orders.anchor, orders.mean_delta, orders.m2) == (2, 1e16, 1.0, 2.0)
    assert (binary.n_success, binary.anchor, binary.mean_delta, binary.m2) == (1, 0.0, 0.5, 0.5)


def test_missing_and_pending_outcomes_do_not_shrink_the_cells(warehouse):
    backend, _, plan, template, bindings, _ = warehouse
    backend.raw_sql(
        """UPDATE outcomes SET finalized_at=NULL WHERE unit_id='001' AND metric='orders';
        DELETE FROM outcomes WHERE unit_id='1' AND metric='orders'"""
    )
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    evidence = capture_overlap_warehouse(plan, capture=capture, **bindings)
    pending, missing = evidence.cells[1:3]
    assert (pending.n_assigned, pending.n_observed, pending.n_pending, pending.anchor) == (
        1,
        0,
        1,
        None,
    )
    assert (missing.n_assigned, missing.n_observed, missing.n_missing, missing.anchor) == (
        1,
        0,
        1,
        None,
    )


def test_guaranteed_admission_refuses_a_negative_gate_even_for_the_enrolled_target(warehouse):
    backend, _, plan, template, bindings, _ = warehouse
    plan = plan.model_copy(update={"target": "jointly_eligible_and_enrolled"})
    backend.raw_sql(
        """UPDATE roster SET gate_b=FALSE WHERE eligible_a AND eligible_b;
        UPDATE decisions SET status='not_enrolled', arm=NULL, assigned_weight=NULL
        WHERE unit_id IN ('001', '1') AND target='target-b'"""
    )
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    with pytest.raises(CodedError) as caught:
        capture_overlap_warehouse(plan, capture=capture, **bindings)
    assert caught.value.code == "overlap.evidence.invalid"
    assert dict(caught.value.context) == {
        "role": "decisions",
        "reason": "admission_count_mismatch",
        "count": 2,
    }


@pytest.mark.parametrize(
    ("sql", "code", "context"),
    [
        (
            "DELETE FROM decisions WHERE unit_id='001' AND target='target-b'",
            "evidence.assignment.invalid",
            {"reason": "missing_decision", "details": ("B", 1)},
        ),
        (
            "UPDATE roster SET eligibility_at_b='2026-01-01 00:00:01+00' WHERE unit_id='001'",
            "overlap.evidence.invalid",
            {"role": "roster", "reason": "eligibility_not_before_either_decision", "count": 1},
        ),
        (
            "UPDATE decisions SET revision='unverified revision' WHERE target='target-b'",
            "overlap.evidence.invalid",
            {"role": "decisions", "reason": "revision_mismatch", "count": 3},
        ),
    ],
)
def test_bad_joins_have_frame_codes(warehouse, sql, code, context):
    backend, _, plan, template, bindings, _ = warehouse
    backend.raw_sql(sql)
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    with pytest.raises(CodedError) as caught:
        capture_overlap_warehouse(plan, capture=capture, **bindings)
    assert caught.value.code == code
    assert dict(caught.value.context) == context


class _CursorCensus:
    def __init__(self, cursor, executed, fetched):
        self._cursor = cursor
        self._executed = executed
        self._fetched = fetched
        self._sql = ""

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *args):
        return self._cursor.__exit__(*args)

    def execute(self, sql, parameters=None):
        self._sql = sql
        self._cursor.execute(sql, parameters)
        if self._cursor.description:
            self._executed(self._columns())
        return self

    def _columns(self):
        return tuple(col[0] for col in self._cursor.description)

    def fetchall(self):
        rows = self._cursor.fetchall()
        self._fetched(self._columns(), rows, self._sql)
        return rows

    def __getattr__(self, key):
        return getattr(self._cursor, key)


def _instrument(backend, monkeypatch, *, executed=lambda _: None, fetched=lambda *_: None):
    def cursor():
        return _CursorCensus(real.cursor(), executed, fetched)

    real = backend.con
    monkeypatch.setattr(backend, "con", _CursorCensus(real, executed, fetched))
    monkeypatch.setattr(backend.con, "cursor", cursor, raising=False)
    return real


def test_query_egress_is_aggregate_only(warehouse, monkeypatch):
    backend, _, plan, template, bindings, _ = warehouse
    executed = []
    materialized = []

    def record(columns, rows, sql):
        if "column_name" in columns:
            assert len(rows) <= 11
        elif "section" in columns:
            allowed = {
                "section",
                "cell",
                "factor",
                "metric",
                "n_without_delivery",
                "n_inconsistent",
                "n_matching",
                "n_assigned",
                "n_observed",
                "n_pending",
                "n_success",
                "anchor",
                "spread",
                "mean_delta",
                "m2",
                *(f"version_{role}" for role in _SCHEMAS),
                *(f"violation_{i}" for i in range(34)),
                *(f"gate_{i}" for i in range(25)),
                *(f"eligibility_{cell}" for cell in ("00", "10", "01", "11")),
            }
            assert set(columns) <= allowed
            assert len(rows) <= 1 + 10 + 4 * len(plan.metrics)
            assert all(
                value is None or isinstance(value, (int, str, float))
                for row in rows
                for value in row
            )
            query = sqlglot.parse_one(sql, dialect="duckdb")
            groups = len(list(query.find_all(exp.Group)))
            joins = sorted(
                {
                    " ".join(filter(None, (join.side, join.kind))) or "INNER"
                    for join in query.find_all(exp.Join)
                }
            )
            materialized.append((len(rows), len(columns), groups, joins))
        else:
            assert set(columns) == {f"version_{role}" for role in _SCHEMAS}
            assert len(rows) == 1

    real = _instrument(backend, monkeypatch, executed=executed.append, fetched=record)
    for units in (3, 1003):
        if units > 3:
            real.execute(
                """INSERT INTO roster SELECT roster.* REPLACE (concat('private-', i) AS unit_id)
                FROM roster CROSS JOIN range(1000) AS n(i) WHERE unit_id='x';
                INSERT INTO decisions SELECT decisions.* REPLACE (concat('private-', i) AS unit_id)
                FROM decisions CROSS JOIN range(1000) AS n(i) WHERE unit_id='x'"""
            )
        capture = bind_overlap_warehouse(plan, capture=template, **bindings)
        assert capture_overlap_warehouse(plan, capture=capture, **bindings).target_count == 2
        print(
            f"DuckDB {duckdb.__version__}, Ibis {ibis.__version__}, {units} units: {materialized[-1]}"
        )
    assert materialized[0] == materialized[1]
    assert materialized[0][0] == 9

    real.execute("INSERT INTO roster SELECT * FROM roster WHERE unit_id='private-826'")
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    with pytest.raises(CodedError) as caught:
        capture_overlap_warehouse(plan, capture=capture, **bindings)
    assert dict(caught.value.context) == {"role": "roster", "reason": "duplicate_key", "count": 1}
    assert "private-826" not in repr(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert sum("section" in columns for columns in executed) == 3
    assert all("unit_id" not in columns and "event_id" not in columns for columns in executed)
    assert materialized[2][0] == 1
    print(f"DuckDB {duckdb.__version__}, Ibis {ibis.__version__}, duplicate: {materialized[2]}")


def test_same_capture_used_for_checks_and_reduction(warehouse, monkeypatch):
    backend, _, plan, template, bindings, _ = warehouse
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    with backend.con.cursor() as writer:
        statements = []

        def write_before_statement(columns, _rows, _sql):
            if "column_name" in columns:
                statements.append(columns)
                if len(statements) == 4:
                    writer.execute("UPDATE outcomes SET value=99 WHERE metric='orders'")

        with monkeypatch.context() as patch:
            _instrument(backend, patch, fetched=write_before_statement)
            original = capture_overlap_warehouse(plan, capture=capture, **bindings)
        assert len(statements) == 4
        assert original.cells[1].anchor == 14.0
        with pytest.raises(CodedError) as caught:
            capture_overlap_warehouse(plan, capture=capture, **bindings)
        assert caught.value.code == "overlap.capture.invalid"
        assert dict(caught.value.context) == {"role": "snapshot", "reason": "unbound_generation"}


def test_source_locator_and_native_snapshot_are_required(warehouse, monkeypatch):
    backend, _, plan, template, bindings, _ = warehouse
    backend.raw_sql(
        """CREATE TABLE roster_other AS SELECT * FROM roster;
        CREATE VIEW roster_view AS SELECT * FROM roster;
        CREATE TABLE roster_volatile (
            study_id VARCHAR, unit_id VARCHAR, entry_at TIMESTAMPTZ,
            eligible_a BOOLEAN GENERATED ALWAYS AS (random() > 0.5),
            eligible_b BOOLEAN, eligibility_at_a TIMESTAMPTZ, eligibility_at_b TIMESTAMPTZ,
            gate_a BOOLEAN, gate_b BOOLEAN, gate_at_a TIMESTAMPTZ, gate_at_b TIMESTAMPTZ
        )"""
    )
    capture = bind_overlap_warehouse(plan, capture=template, **bindings)
    executed = []
    _instrument(backend, monkeypatch, executed=executed.append)
    for table in ("roster_other", "roster_view"):
        changed = {**bindings, "roster": backend.table(table, database=("overlap", "main"))}
        with pytest.raises(CodedError) as caught:
            capture_overlap_warehouse(plan, capture=capture, **changed)
        assert caught.value.code == "overlap.capture.invalid"
        assert dict(caught.value.context) == {"role": "snapshot", "reason": "unbound_generation"}
    assert not any("section" in cols for cols in executed)
    volatile = {
        **bindings,
        "roster": backend.table("roster_volatile", database=("overlap", "main")),
    }
    with pytest.raises(CodedError) as caught:
        bind_overlap_warehouse(plan, capture=template, **volatile)
    assert dict(caught.value.context) == {"role": "snapshot", "reason": "unbound_generation"}
