"""Shared, dialect-parametrized probes reused by every real-backend
warehouse execution test. Each function drives the actual PUBLIC
Analysis.from_definitions entry point end to end -- never a
hand-assembled query_builders-only pipeline -- so a real backend's
source binding, moments-source dispatch, cache/materialization, and
artifact publish/adopt code all execute for real. Expected values are
the corrected W01/W02/W03 contracts, not the reviewed baseline defects."""

from __future__ import annotations

import datetime as dt
import tempfile
from contextlib import ExitStack
from pathlib import Path

from increment import Analysis
from increment.query.artifact_contract import unit_day_artifact_extension_catalog
from increment.query.session import WarehouseArtifactStore


def _postgres_temp_schema(con):
    with con.raw_sql(
        "SELECT nspname FROM pg_catalog.pg_namespace WHERE oid = pg_my_temp_schema()"
    ) as cursor:
        row = cursor.fetchone()
    assert row is not None and row[0].startswith("pg_temp_"), "missing physical TEMP schema"
    return row[0]


def _physical_temp_relation(con, relation, dialect):
    if dialect == "postgres":
        return con.table(relation.op().name, database=_postgres_temp_schema(con))
    if dialect == "snowflake":
        namespace = relation.op().namespace
        catalog = namespace.catalog or con.current_catalog
        database = namespace.database or con.current_database
        assert catalog and database, "missing physical TEMP namespace"
        return con.table(relation.op().name, database=(catalog, database))
    return relation


def _materialization_database(relation):
    namespace = relation.op().namespace
    return (
        (namespace.catalog, namespace.database)
        if namespace.catalog is not None and namespace.database is not None
        else namespace.database or namespace.catalog
    )


def _required_materializations(con, created):
    required = []
    for role, prefix in (("spine", "exp_spine_"), ("revenue stats", "exp_stats_revenue_")):
        relations = [relation for name, relation in created if name.startswith(prefix)]
        assert relations, f"store='always' did not materialize {role}"
        for relation in relations:
            database = _materialization_database(relation)
            assert relation.op().name in con.list_tables(database=database), (
                f"materialization not catalog-visible: {database!r}.{relation.op().name}"
            )
            assert int(relation.count().execute()) > 0, role
        required.extend(relations)
    return tuple(required)


def _assert_materializations_absent(con, relations):
    for relation in relations:
        op = relation.op()
        database = _materialization_database(relation)
        assert op.name not in con.list_tables(database=database), (
            f"materialization leaked: {database!r}.{op.name}"
        )


def _quoted_identifier(dialect: str, name: str) -> str:
    quote = "`" if dialect == "bigquery" else '"'
    return f"{quote}{name}{quote}"


def _qualified_table_identifier(con, dialect: str, name: str) -> str:
    """A table reference usable in raw SQL.

    BigQuery resolves nothing from session state, so a bare table name in a
    fact source or a DML statement is rejected outright; it needs the project
    and dataset. The other dialects resolve an unqualified name themselves.
    """
    if dialect != "bigquery":
        return _quoted_identifier(dialect, name)
    catalog = con.current_catalog
    database = con.current_database
    assert catalog and database, "missing BigQuery namespace"
    return ".".join(_quoted_identifier(dialect, part) for part in (catalog, database, name))


def _cache_correction_sql(con, dialect: str) -> str:
    table = _qualified_table_identifier(con, dialect, "events_cache_smoke")
    value = _quoted_identifier(dialect, "value")
    event = _quoted_identifier(dialect, "event")
    unit_id = _quoted_identifier(dialect, "user_id")
    return (
        f"UPDATE {table} SET {value} = {value} * 2 "
        f"WHERE {event} = 'purchase' AND {unit_id} LIKE 't%'"
    )


def run_ratio_metric_probe(con, dialect: str) -> None:
    """The avg_event/count ratio returns the corrected +25% lift through
    Analysis.from_definitions. Backend aggregation order may differ in
    the last few bits, so assert the independently derived value within
    tolerance rather than byte identity."""
    source = _qualified_table_identifier(con, dialect, "events_ratio_smoke")
    yaml = f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: enrolled, column: null}}
      - {{name: num, column: value}}
      - {{name: den, column: null}}
exposures:
  - {{name: enrollment, fact: enrolled}}
metrics:
  - name: ratio
    type: ratio
    entity: user_id
    numerator: {{fact: num, aggregation: avg_event, window_days: 2}}
    denominator: {{fact: den, aggregation: count, window_days: 2}}
  - {{name: conversion, type: conversion, entity: user_id, fact: num, window_days: 2}}
  - {{name: mean, type: mean, entity: user_id, fact: num, aggregation: sum, window_days: 2}}
  - {{name: retention, type: retention, entity: user_id, fact: num, threshold_days: [1, 2]}}
  - {{name: quantile, type: quantile, entity: user_id, fact: num, aggregation: sum, quantile: 0.25}}
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-01-03T00:00:00
    control_group: control
    plan: {{secondaries: [ratio, conversion, mean, retention, quantile]}}
"""
    rows = []
    for arm, prefix in [("control", "c"), ("treatment", "t")]:
        for i in range(30):
            rows.append(
                {
                    "user_id": f"{prefix}{i}",
                    "ts": dt.datetime(2025, 1, 1, 8),
                    "event": "enrolled",
                    "experiment_id": "exp",
                    "group_id": arm,
                    "value": None,
                }
            )
            value = (0.375 if arm == "control" else 1.375) + i / 4
            values = [value, value] if arm == "control" else [value]
            for j, value in enumerate(values):
                rows.append(
                    {
                        "user_id": f"{prefix}{i}",
                        "ts": dt.datetime(2025, 1, 2, 9, j),
                        "event": "num",
                        "experiment_id": None,
                        "group_id": None,
                        "value": value,
                    }
                )
            rows.append(
                {
                    "user_id": f"{prefix}{i}",
                    "ts": dt.datetime(2025, 1, 2, 11),
                    "event": "den",
                    "experiment_id": None,
                    "group_id": None,
                    "value": None,
                }
            )
    con.create_table("events_ratio_smoke", obj=rows, overwrite=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "defs.yaml"
            path.write_text(yaml)
            analysis = Analysis.from_definitions("exp", path, con, store="none")
            try:
                row = next(row for row in analysis.run() if row.metric == "ratio")
                assert abs(row.lift.value - 0.25) < 1e-9, (
                    f"expected +25% ratio lift, got {row.lift.value}"
                )
                groups = analysis.dashboard_group_data(metrics=analysis.metrics)
                # Linear q=.25 uses zero-based rank 7.25 for 30 units.
                # Unit-total ramps are 0.75 + i/2 and 1.375 + i/4, respectively.
                expected = {
                    "control": {
                        "ratio": 4.0,
                        "conversion": 1.0,
                        "mean": 8.0,
                        "retention": 1.0,
                        "quantile": 4.375,
                    },
                    "treatment": {
                        "ratio": 5.0,
                        "conversion": 1.0,
                        "mean": 5.0,
                        "retention": 1.0,
                        "quantile": 3.1875,
                    },
                }
                assert {(group.metric, group.group_id) for group in groups} == {
                    (metric, arm) for arm, values in expected.items() for metric in values
                }
                for group in groups:
                    assert abs(group.observed_value - expected[group.group_id][group.metric]) < 1e-9
                    assert group.assigned_units == group.eligible_units == 30
                    assert (
                        group.excluded_not_mature
                        == group.excluded_no_observed_day
                        == group.excluded_other
                        == 0
                    )
                    assert group.event_count == (60 if group.group_id == "control" else 30)
                    if group.metric == "ratio":
                        assert (
                            abs(group.numerator - (120 if group.group_id == "control" else 150))
                            < 1e-9
                        )
                        assert group.denominator == 30
                    if group.metric in ("conversion", "retention"):
                        assert group.retained_units == 30
            finally:
                analysis.close()
    finally:
        con.drop_table("events_ratio_smoke", force=True)


def _materialized_cache_fixture(con, dialect: str) -> tuple[str, list[dict[str, object]]]:
    """Shared enrollment and revenue fixture for materialization lifecycle probes."""
    source = _qualified_table_identifier(con, dialect, "events_cache_smoke")
    yaml = f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source}
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: enrolled, column: null}}
      - {{name: purchase, column: value}}
exposures:
  - {{name: enrollment, fact: enrolled}}
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 2
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-01-03T00:00:00
    control_group: control
    plan: {{secondaries: [revenue]}}
"""
    rows = []
    for arm, prefix, base in [("control", "c", 10.0), ("treatment", "t", 12.0)]:
        for i in range(30):
            rows.append(
                {
                    "user_id": f"{prefix}{i}",
                    "ts": dt.datetime(2025, 1, 1, 8),
                    "event": "enrolled",
                    "experiment_id": "exp",
                    "group_id": arm,
                    "value": None,
                }
            )
            rows.append(
                {
                    "user_id": f"{prefix}{i}",
                    "ts": dt.datetime(2025, 1, 2, 9),
                    "event": "purchase",
                    "experiment_id": None,
                    "group_id": None,
                    "value": base + i % 3,
                }
            )
    return yaml, rows


def run_materialized_cache_correction_probe(con, dialect: str, monkeypatch) -> None:
    """Check fresh materialized results, physical relations, and cleanup."""
    yaml, rows = _materialized_cache_fixture(con, dialect)
    con.create_table("events_cache_smoke", obj=rows, overwrite=True)
    created_relations = []
    create_table = con.create_table

    def recording_create_table(name, *args, **kwargs):
        relation = create_table(name, *args, **kwargs)
        if kwargs.get("temp") is True:
            physical = _physical_temp_relation(con, relation, dialect)
            created_relations.append((name, physical))
        return relation

    monkeypatch.setattr(con, "create_table", recording_create_table)
    analysis = None
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "defs.yaml"
            path.write_text(yaml)
            analysis = Analysis.from_definitions("exp", path, con, store="always")
            before = analysis.run()[0].lift.value
            first_materializations = _required_materializations(con, created_relations)
            first_creation_count = len(created_relations)
            con.raw_sql(_cache_correction_sql(con, dialect))
            cached = analysis.run()[0].lift.value
            second_materializations = _required_materializations(
                con, created_relations[first_creation_count:]
            )
            _assert_materializations_absent(con, first_materializations)
            current = Analysis.from_definitions("exp", path, con, store="none").run()[0].lift.value
            assert abs(before - (2 / 11)) < 1e-9, f"expected pre-correction lift 2/11, got {before}"
            assert abs(cached - (15 / 11)) < 1e-9, (
                f"expected post-correction lift 15/11, got {cached}"
            )
            assert abs(current - (15 / 11)) < 1e-9, (
                f"independent live read was incorrect: {current}"
            )
            assert abs(current - cached) < 1e-9, (
                f"materialized re-read stayed stale: {cached} vs {current}"
            )
            analysis.close()
            _assert_materializations_absent(con, second_materializations)
    finally:
        if analysis is not None:
            analysis.close()
        con.drop_table("events_cache_smoke", force=True)


def _drop_probe_generation(
    con, store, ref, dialect: str, *, schema_name: str | None = None
) -> None:
    store.drop_generation(ref.artifact_id, ref.generation_id)
    # The probe has no external readers; remove its retained receipts as well.
    for name in ("ud_manifest_index", "ud_manifest_dropped"):
        table = (
            f"{_quoted_identifier(dialect, schema_name)}.{_quoted_identifier(dialect, name)}"
            if schema_name is not None
            else _qualified_table_identifier(con, dialect, name)
        )
        artifact = _quoted_identifier(dialect, "artifact_id")
        generation = _quoted_identifier(dialect, "generation_id")
        con.raw_sql(
            f"DELETE FROM {table} WHERE {artifact} = '{ref.artifact_id}' "
            f"AND {generation} = '{ref.generation_id}'"
        )


_SECRET = {
    "fact_used": "SECRET_FACT_USED_5b1e",
    "fact_unused": "SECRET_FACT_UNUSED_77c2",
    "dim_used": "SECRET_DIM_USED_a90d",
    "dim_unused": "SECRET_DIM_UNUSED_e4c1",
    "exposure_used": "SECRET_EXPOSURE_USED_31fe",
    "exposure_unused": "SECRET_EXPOSURE_UNUSED_9d02",
}
_MAIN_PROBE_SECRETS = ("fact_used", "fact_unused", "dim_unused", "exposure_unused")


def _assert_artifact_output_confidential(
    con, store, ref, context, names, *, planted_in: str
) -> None:
    """No planted SQL sentinel reaches the ref, context, manifest or persisted index row."""
    secrets = {name: _SECRET[name] for name in names}
    for name, sentinel in secrets.items():
        assert sentinel in planted_in, f"probe never planted the {name} sentinel"
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    table = con.table("ud_manifest_index")
    persisted = [
        row
        for row in table.to_pyarrow().to_pylist()
        if row["artifact_id"] == str(ref.artifact_id)
        and row["generation_id"] == str(ref.generation_id)
    ]
    assert len(persisted) == 1, "published manifest index row is not readable back"
    assert persisted[0]["manifest_json"], "persisted manifest_json is empty"
    surfaces = {
        "ref": repr(ref) + ref.model_dump_json(),
        "context": context.canonical_json,
        "manifest": manifest.model_dump_json() + repr(manifest),
        "manifest_index_row": "".join(str(value) for value in persisted[0].values()),
    }
    for surface, text in surfaces.items():
        for name, sentinel in secrets.items():
            assert sentinel not in text, f"{name} SQL sentinel leaked into the {surface}"


def _scratch_catalog(con, dialect: str) -> str | None:
    """The project or database a hosted backend needs to qualify a scratch namespace."""
    if dialect not in ("snowflake", "bigquery"):
        return None
    catalog = con.current_catalog
    assert catalog, f"missing {dialect} catalog for the scratch namespace"
    return catalog


def _create_scratch_namespace(con, dialect: str, schema: str) -> None:
    catalog = _scratch_catalog(con, dialect)
    if catalog is None:
        con.create_database(schema)
    else:
        con.create_database(schema, catalog=catalog)


def _drop_scratch_namespace(con, schema: str, catalog: str | None = None) -> None:
    """Drop a probe-owned namespace and its tables (not every backend cascades).

    Tables are addressed by the bare schema name: hosted ``create_table`` accepts
    only a string database, and the scratch catalog is the connection's current
    one, so the bare name resolves to the same namespace.
    """
    for name in con.list_tables(database=schema):
        con.drop_table(name, database=schema, force=True)
    if catalog is None:
        con.drop_database(schema, force=True)
    else:
        con.drop_database(schema, catalog=catalog, force=True)


def _assert_boolean_dimension_labels(
    native: Analysis, adopted: Analysis, *, clustered: bool
) -> None:
    """A genuine BOOLEAN dimension with one NULL unit labels like the native path.

    Only the unclustered design declares the ``is_vip`` breakout.
    """
    if clustered:
        return
    for name, analysis in (("native", native), ("adopted", adopted)):
        labels = {row.dimension_value for row in analysis.run_daily(dimension="is_vip")}
        assert labels == {"true", "false", "__null__"}, (name, labels)


def run_artifact_publish_adopt_probe(con, dialect: str, *, clustered: bool = True) -> None:
    """Publish and adopt a unit-day artifact on the live backend.

    The clustered and unclustered variants use separate valid experiment
    designs: a clustered experiment requests cluster identity, while the
    unclustered experiment requests a breakout dimension. Both must preserve
    the native -50% lift and 60/60 arm counts.
    """
    experiment_identity = (
        "    cluster: cluster_id\n"
        if clustered
        else "    breakouts:\n      - {property: cluster_label}\n      - {property: is_vip}\n"
    )
    property_declaration = (
        ""
        if clustered
        else "    properties:\n"
        "      - {name: cluster_label, column: cluster_id, dtype: string, as_of: static}\n"
        "      - {name: is_vip, column: is_vip, dtype: bool, as_of: static}\n"
    )
    requested_kinds = (
        {"cluster_identity", "assignment_counts"}
        if clustered
        else {"assignment_counts", "breakout_dimension"}
    )
    source = _qualified_table_identifier(con, dialect, "events_artifact_smoke")
    yaml = f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {source} /* {_SECRET["fact_used"]} */
    timestamp_column: ts
    entities: [user_id, cluster_id]
{property_declaration}    facts:
      - {{name: enrolled, column: null}}
      - {{name: purchase, column: value}}
  - name: unused_events
    sql: SELECT * FROM {source} /* {_SECRET["fact_unused"]} */
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: unused_fact, column: null}}
dim_sources:
  - name: unused_users
    sql: SELECT * FROM {source} /* {_SECRET["dim_unused"]} */
    entity: user_id
    properties:
      - {{name: unused_segment, column: cluster_id, dtype: string, as_of: static}}
exposures:
  - {{name: enrollment, fact: enrolled}}
  - name: unused_direct
    sql: SELECT user_id AS unit_id, ts, group_id FROM {source} /* {_SECRET["exposure_unused"]} */
metrics:
  - name: conversion
    type: conversion
    entity: user_id
    fact: purchase
    window_days: 3
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
{experiment_identity}    plan: {{secondaries: [conversion]}}
"""
    rows = []
    for arm, prefix in [("control", "c"), ("treatment", "t")]:
        for i in range(60):
            exposure_day = 9 if arm == "treatment" and i >= 30 else 1
            uid = f"{prefix}{i}"
            cluster_id = None if not clustered and i == 0 else uid
            rows.append(
                {
                    "user_id": uid,
                    "cluster_id": cluster_id,
                    "is_vip": None if i == 0 else i % 2 == 0,
                    "ts": dt.datetime(2025, 1, exposure_day, 8),
                    "event": "enrolled",
                    "experiment_id": "exp",
                    "group_id": arm,
                    "value": None,
                }
            )
            converted = i < 30 if arm == "control" else i < 15
            if converted:
                rows.append(
                    {
                        "user_id": uid,
                        "cluster_id": cluster_id,
                        "is_vip": None if i == 0 else i % 2 == 0,
                        "ts": dt.datetime(2025, 1, 2, 9),
                        "event": "purchase",
                        "experiment_id": None,
                        "group_id": None,
                        "value": 1.0,
                    }
                )
    rows.append(
        {
            "user_id": "ghost",
            "cluster_id": "ghost",
            "is_vip": True,
            "ts": dt.datetime(2025, 1, 12, 8),
            "event": "purchase",
            "experiment_id": None,
            "group_id": None,
            "value": 1.0,
        }
    )
    with ExitStack() as cleanup:
        con.create_table("events_artifact_smoke", obj=rows, overwrite=True)
        cleanup.callback(con.drop_table, "events_artifact_smoke", force=True)
        store = WarehouseArtifactStore(con)
        with ExitStack() as analyses:
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "defs.yaml"
                path.write_text(yaml)
                live = Analysis.from_definitions("exp", path, con, store="none")
                analyses.callback(live.close)
                before = live.run()[0]
                context = live._artifact_context(
                    live._defs, live._experiment, live._on_mixed_assignment
                )
                requests = [
                    e.request
                    for e in unit_day_artifact_extension_catalog(context)
                    if e.request.kind in requested_kinds
                ]
                ref = live.publish_unit_day_artifact(store, extensions=requests)
                cleanup.callback(_drop_probe_generation, con, store, ref, dialect)
                adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
                analyses.callback(adopted.close)
                after = adopted.run()[0]
                assert abs(before.lift.value - (-0.5)) < 1e-9, (
                    f"native artifact fixture lift was not -50%: {before.lift.value}"
                )
                assert abs(after.lift.value - (-0.5)) < 1e-9, (
                    f"adopted artifact fixture lift was not -50%: {after.lift.value}"
                )
                assert abs(before.lift.value - after.lift.value) < 1e-9
                assert abs(before.lift.lb - after.lift.lb) < 1e-9
                assert abs(before.lift.ub - after.lift.ub) < 1e-9
                _assert_boolean_dimension_labels(live, adopted, clustered=clustered)
                for counted in (live, adopted):
                    assert counted._src.unit_counts() == {"control": 60, "treatment": 60}
                with store.open_snapshot(ref) as snapshot:
                    manifest = snapshot.read_manifest(
                        ref.manifest, expected_sha256=ref.manifest_sha256
                    )
                for relation_ref in (manifest.base.exposures, manifest.base.measure_stats):
                    stored = con.table(
                        relation_ref.relation.name,
                        database=relation_ref.relation.schema_name,
                    )
                    assert int(stored.count().execute()) == relation_ref.row_count
                exposure_locator = manifest.base.exposures.relation
                exposure_ids = {
                    row["unit_id"]
                    for row in con.table(
                        exposure_locator.name, database=exposure_locator.schema_name
                    )
                    .select("unit_id")
                    .to_pyarrow()
                    .to_pylist()
                }
                extension_kinds = {extension.kind for extension in manifest.extensions}
                assert requested_kinds <= extension_kinds
                for extension in manifest.extensions:
                    if extension.kind not in ("breakout_dimension", "factor_dimension"):
                        continue
                    locator = extension.relation.relation
                    dimension_rows = (
                        con.table(locator.name, database=locator.schema_name)
                        .select("unit_id", "value_is_missing")
                        .to_pyarrow()
                        .to_pylist()
                    )
                    keys = [row["unit_id"] for row in dimension_rows]
                    assert len(keys) == len(set(keys))
                    assert set(keys) == exposure_ids
                    assert "ghost" not in keys
                    if not clustered:
                        missing = {
                            row["unit_id"] for row in dimension_rows if row["value_is_missing"]
                        }
                        assert "c0" in missing
                _assert_artifact_output_confidential(
                    con, store, ref, context, _MAIN_PROBE_SECRETS, planted_in=yaml
                )


def run_artifact_confidentiality_probe(con, dialect: str) -> None:
    """A published artifact never carries source SQL, used or not.

    Unique sentinels sit in the fact, dimension and direct-exposure SQL that the
    experiment reads AND in a second fact source, dimension source and exposure
    it never references. The published ref, the caller's context, the manifest
    and the manifest JSON persisted in the warehouse index must all be free of
    them, while the adopted artifact still reproduces the native readout.
    """
    secrets = _SECRET
    events = _qualified_table_identifier(con, dialect, "events_confidentiality_smoke")
    users = _qualified_table_identifier(con, dialect, "users_confidentiality_smoke")
    enrolled = _qualified_table_identifier(con, dialect, "enrolled_confidentiality_smoke")
    unit = _quoted_identifier(dialect, "user_id")
    unit_alias = _quoted_identifier(dialect, "unit_id")
    timestamp = _quoted_identifier(dialect, "ts")
    group = _quoted_identifier(dialect, "group_id")
    yaml = f"""
dialect: {dialect}
fact_sources:
  - name: events
    sql: SELECT * FROM {events} /* {secrets["fact_used"]} */
    timestamp_column: ts
    entities: [user_id]
    dims: [users]
    facts:
      - {{name: purchase, column: null}}
  - name: unused_events
    sql: SELECT * FROM {events} /* {secrets["fact_unused"]} */
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {{name: unused_fact, column: null}}
dim_sources:
  - name: users
    sql: SELECT * FROM {users} /* {secrets["dim_used"]} */
    entity: user_id
    properties:
      - {{name: segment, column: segment, dtype: string, as_of: static}}
  - name: unused_users
    sql: SELECT * FROM {users} /* {secrets["dim_unused"]} */
    entity: user_id
    properties:
      - {{name: unused_segment, column: segment, dtype: string, as_of: static}}
exposures:
  - name: enrollment
    sql: SELECT {unit} AS {unit_alias}, {timestamp}, {group} FROM {enrolled} /* {secrets["exposure_used"]} */
  - name: unused_enrollment
    sql: SELECT {unit} AS {unit_alias}, {timestamp}, {group} FROM {enrolled} /* {secrets["exposure_unused"]} */
metrics:
  - name: conversion
    type: conversion
    entity: user_id
    fact: purchase
    window_days: 3
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    breakouts:
      - {{property: segment}}
    plan: {{secondaries: [conversion]}}
"""
    enrollments, purchases, dimension_rows = [], [], []
    for arm, prefix in [("control", "c"), ("treatment", "t")]:
        for i in range(40):
            uid = f"{prefix}{i}"
            enrollments.append({"user_id": uid, "ts": dt.datetime(2025, 1, 1, 8), "group_id": arm})
            dimension_rows.append({"user_id": uid, "segment": "a" if i % 2 else "b"})
            if i < (20 if arm == "control" else 10):
                purchases.append(
                    {"user_id": uid, "ts": dt.datetime(2025, 1, 2, 9), "event": "purchase"}
                )
    # A post-window event from an unenrolled unit closes the data horizon.
    purchases.append({"user_id": "ghost", "ts": dt.datetime(2025, 1, 12, 8), "event": "purchase"})
    with ExitStack() as cleanup:
        for name, table_rows in (
            ("events_confidentiality_smoke", purchases),
            ("users_confidentiality_smoke", dimension_rows),
            ("enrolled_confidentiality_smoke", enrollments),
        ):
            con.create_table(name, obj=table_rows, overwrite=True)
            cleanup.callback(con.drop_table, name, force=True)
        store = WarehouseArtifactStore(con)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "defs.yaml"
            path.write_text(yaml)
            live = Analysis.from_definitions("exp", path, con, store="none")
            cleanup.callback(live.close)
            before = live.run()[0]
            context = live._artifact_context(
                live._defs, live._experiment, live._on_mixed_assignment
            )
            requests = [
                e.request
                for e in unit_day_artifact_extension_catalog(context)
                if e.request.kind in {"breakout_dimension", "assignment_counts"}
            ]
            ref = live.publish_unit_day_artifact(store, extensions=requests)
            cleanup.callback(_drop_probe_generation, con, store, ref, dialect)
            adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
            cleanup.callback(adopted.close)
            after = adopted.run()[0]
            assert abs(before.lift.value - (-0.5)) < 1e-9, before.lift.value
            assert abs(before.lift.value - after.lift.value) < 1e-9
            assert abs(before.lift.lb - after.lift.lb) < 1e-9
            assert abs(before.lift.ub - after.lift.ub) < 1e-9
            _assert_artifact_output_confidential(
                con, store, ref, context, tuple(secrets), planted_in=yaml
            )


def _sequential_definitions(dialect: str, source: str, *, registered: bool):
    """Definitions for the shared binary sequential fixture.

    ``registered=False`` keeps the plan's automatic always-valid registration.
    ``registered=True`` replaces it with an explicit Bernoulli registration bound
    to the same definitions, so the artifact reader must adopt a caller-declared
    plan rather than re-derive one.
    """
    from increment import SequentialRegistration, sequential_definition_id
    from increment.semantics.loader import load
    from increment.semantics.models import AnalysisPlan, InferenceSpec
    from increment.sequential_source import native_observation_mapping
    from tests.binary_sequential_cases import ROUTES, _definitions_file
    from tests.sequential_cases import registration

    with tempfile.TemporaryDirectory() as td:
        definitions = load(
            str(_definitions_file(td, dialect, source, ROUTES["always_valid"].plan_yaml))
        )
    if not registered:
        return definitions
    experiment = definitions.experiments[0]
    base = registration("bernoulli").model_dump()
    base["source_id"] = experiment.name
    base["models"] = [{**base["models"][0], "metric": "purchase"}]
    base["roster"] = [{**base["roster"][0], "metric": "purchase"}]
    base["reveal"] = {**base["reveal"], "longest_window_days": 2}
    base["definitions_id"] = sequential_definition_id(
        definitions.metrics,
        experiment.resolved_design(),
        source_mapping=native_observation_mapping(definitions, experiment),
    )
    plan = AnalysisPlan(
        primary="purchase",
        inference=InferenceSpec(
            kind="always_valid", registration=SequentialRegistration.model_validate(base)
        ),
    )
    experiment = experiment.model_copy(update={"plan": plan})
    return definitions.model_copy(update={"experiments": (experiment,)})


def _close_to(left, right, *, path="value") -> None:
    """Equal structure; floats within 1e-9 relative/absolute, everything else exact."""
    if isinstance(left, float) or isinstance(right, float):
        assert left is not None and right is not None, (path, left, right)
        assert abs(left - right) <= 1e-9 * max(1.0, abs(left), abs(right)), (path, left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), (path, left, right)
        for key in left:
            _close_to(left[key], right[key], path=f"{path}.{key}")
    elif isinstance(left, list | tuple):
        assert isinstance(right, list | tuple) and len(left) == len(right), (path, left, right)
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _close_to(a, b, path=f"{path}[{i}]")
    else:
        assert left == right, (path, left, right)


def _assert_same_sequential_readout(native: Analysis, adopted: Analysis) -> None:
    """Adopted equals native: registration, retained state, moments and interval."""
    from tests.binary_sequential_cases import AS_OF
    from tests.parity_harness.runner import sequential_fingerprint

    native_snapshot = native.capture_sequential(finalized=True, as_of=AS_OF)
    adopted_snapshot = adopted.capture_sequential(finalized=True, as_of=AS_OF)
    assert (
        native_snapshot.registration.definitions_id == adopted_snapshot.registration.definitions_id
    )
    assert native_snapshot.registration == adopted_snapshot.registration
    # prefix_id depends on unit-proof accumulation order, which publish/reopen does not
    # preserve; compare arm states and the unit proofs as an order-independent set instead.
    assert sequential_fingerprint(native_snapshot) == sequential_fingerprint(adopted_snapshot)
    assert sorted((r.unit_id, r.group_id, r.digest) for r in native_snapshot.records) == sorted(
        (r.unit_id, r.group_id, r.digest) for r in adopted_snapshot.records
    )
    native_moments = native._src.moments(
        native._src.context.metrics[0], grain="asof", completed_windows_only=True
    )
    adopted_moments = adopted._src.moments(
        adopted._src.context.metrics[0], grain="asof", completed_windows_only=True
    )
    assert native_moments, "native sequential fixture produced no moments"
    _close_to(
        sorted(native_moments, key=lambda row: (str(row["ds"]), row["metric"], row["group_id"])),
        sorted(adopted_moments, key=lambda row: (str(row["ds"]), row["metric"], row["group_id"])),
        path="moments",
    )
    native_lift = native.run()[0].require_lift()
    adopted_lift = adopted.run()[0].require_lift()
    assert native_lift.open_side == adopted_lift.open_side
    assert native_lift.level == adopted_lift.level
    for field in ("value", "lb", "ub"):
        _close_to(getattr(native_lift, field), getattr(adopted_lift, field), path=field)


def _sequential_fixture_table(con, dialect: str, name: str) -> str:
    from tests.binary_sequential_cases import event_rows, unit_rows

    con.create_table(name, obj=event_rows(unit_rows()), overwrite=True)
    return _qualified_table_identifier(con, dialect, name)


def run_artifact_sequential_identity_probe(con, dialect: str) -> None:
    """An adopted artifact is the same sequential study as its native capture.

    For the automatic asymptotic and always-valid plans and for an explicitly
    registered plan, the adopted artifact reports the same registration and
    ``sequential_definition_id``, the same retained state, the same daily
    moments and the same interval as the native definitions source.
    """
    from increment.query.artifact_publish import artifact_context
    from tests.analysis_factory import make_analysis
    from tests.binary_sequential_cases import (
        ROUTES,
        expected_artifact_context,
        warehouse_analysis,
    )

    table = "events_seqid_smoke"
    with ExitStack() as cleanup:
        source = _sequential_fixture_table(con, dialect, table)
        cleanup.callback(con.drop_table, table, force=True)
        store = WarehouseArtifactStore(con)
        cases = [(ROUTES[kind].plan_yaml, None) for kind in ("asymptotic_mean", "always_valid")]
        cases.append((None, _sequential_definitions(dialect, source, registered=True)))
        for plan_yaml, registered_definitions in cases:
            if registered_definitions is None:
                native = warehouse_analysis(con, dialect, source, plan_yaml)
                context = expected_artifact_context(dialect, source, plan_yaml)
            else:
                experiment = registered_definitions.experiments[0]
                native = make_analysis(con, registered_definitions, experiment=experiment)
                context = artifact_context(registered_definitions, experiment, "error")
            with ExitStack() as case:
                case.callback(native.close)
                ref = native.publish_unit_day_artifact(store)
                case.callback(_drop_probe_generation, con, store, ref, dialect)
                adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
                case.callback(adopted.close)
                _assert_same_sequential_readout(native, adopted)


def _create_index_table(con, schema: str, columns: dict[str, str]) -> None:
    import ibis

    con.create_table("ud_manifest_index", schema=ibis.schema(columns), database=schema)


def run_artifact_namespace_probe(con, dialect: str) -> None:
    """A pre-created manifest index is read and written by column name.

    A namespace whose ``ud_manifest_index`` carries the all-string columns in
    REVERSED physical order publishes, and a fresh store over the same namespace
    adopts the published ref to the native readout. A namespace whose index has a
    wrong-typed or a missing column refuses at store construction with
    ``artifact.store.namespace``.
    """
    import json
    from uuid import uuid4

    from increment.errors import CapabilityError
    from tests.binary_sequential_cases import ROUTES, expected_artifact_context, warehouse_analysis

    index_columns = WarehouseArtifactStore._INDEX_TABLE_COLUMNS
    reversed_columns = dict.fromkeys(reversed(index_columns), "string")
    plan_yaml = ROUTES["always_valid"].plan_yaml
    table = "events_namespace_smoke"
    catalog = _scratch_catalog(con, dialect)
    with ExitStack() as cleanup:
        source = _sequential_fixture_table(con, dialect, table)
        cleanup.callback(con.drop_table, table, force=True)

        def scratch_namespace() -> str:
            schema = f"ud_probe_{uuid4().hex[:12]}"
            _create_scratch_namespace(con, dialect, schema)
            cleanup.callback(_drop_scratch_namespace, con, schema, catalog)
            return schema

        reordered = scratch_namespace()
        _create_index_table(con, reordered, reversed_columns)
        assert tuple(con.table("ud_manifest_index", database=reordered).columns) == tuple(
            reversed_columns
        ), "backend did not preserve the reversed physical column order"
        native = warehouse_analysis(con, dialect, source, plan_yaml)
        cleanup.callback(native.close)
        context = expected_artifact_context(dialect, source, plan_yaml)
        ref = native.publish_unit_day_artifact(WarehouseArtifactStore(con, schema_name=reordered))
        # A fresh store keeps no in-process generation: the ref resolves only
        # through what the reversed physical index persisted.
        fresh = WarehouseArtifactStore(con, schema_name=reordered)
        index_rows = con.table("ud_manifest_index", database=reordered).to_pyarrow().to_pylist()
        (row,) = [r for r in index_rows if r["artifact_id"] == str(ref.artifact_id)]
        assert row["generation_id"] == str(ref.generation_id)
        assert row["manifest_name"] == ref.manifest.name
        assert row["manifest_sha256"] == ref.manifest_sha256
        with fresh.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        assert json.loads(row["manifest_json"]) == manifest.model_dump(mode="json")
        adopted = Analysis.from_unit_day_artifact(fresh, ref, expected_context=context)
        cleanup.callback(adopted.close)
        _assert_same_sequential_readout(native, adopted)

        for label, broken in (
            ("wrong_type", {**dict.fromkeys(index_columns, "string"), "manifest_json": "int64"}),
            (
                "missing_column",
                {c: "string" for c in index_columns if c != "manifest_sha256"},
            ),
        ):
            namespace = scratch_namespace()
            _create_index_table(con, namespace, broken)
            try:
                WarehouseArtifactStore(con, schema_name=namespace)
            except CapabilityError as error:
                assert error.code == "artifact.store.namespace", (label, error.code)
            else:
                raise AssertionError(f"{label} manifest index was accepted by the store")


def _publish_then_abort(store, after_manifest) -> None:
    """Open a publication, write both base relations, publish a minimal manifest, then
    call ``after_manifest(ref, exposure_ref, measure_ref)`` inside the open publication."""
    import hashlib

    import ibis

    from increment.query.artifact_contract import ARTIFACT_DOMAIN_ROOT
    from increment.query.artifact_digest import manifest_sha256
    from increment.semantics.artifact import (
        ArtifactContext,
        BaseRelations,
        MeasureManifest,
        UnitDayArtifactManifest,
    )

    payload = (
        '{"context_format":2,"experiment_name":"demo",'
        '"window_days":{"end":null,"observation_horizon":null,"start":"2025-01-01"}}'
    )
    context = ArtifactContext(
        canonical_json=payload,
        sha256=hashlib.sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + payload.encode()).hexdigest(),
    )
    day = dt.date(2024, 1, 1)
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "group_id": "control",
                        "first_exposure_ts": dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
                        "first_exposure_date": day,
                    }
                ]
            ),
        )
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "ds": day,
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        body = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": "demo",
            "created_at": dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            "day_boundary": "UTC",
            "first_ds": day,
            "last_ds": day,
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": day, "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
        }
        prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
        body["manifest_sha256"] = manifest_sha256(prototype)
        ref = publication.publish_manifest(UnitDayArtifactManifest(**body))
        after_manifest(ref, exposure_ref, measure_ref)


def run_artifact_abort_probe(con, dialect: str) -> None:
    """A publication aborted after its manifest is published leaves nothing visible.

    The body raises right after ``publish_manifest`` against a probe-owned
    namespace. The exception propagates unchanged, a FRESH store over the same
    namespace (no in-process generation) lists no visible manifest, opening the
    published ref refuses ``artifact.generation.dropped``, and both written
    relations are gone from the warehouse.
    """
    from uuid import uuid4

    import pytest

    from increment.query.artifact_contract import ArtifactContractError

    class _BodyFailed(Exception):
        pass

    failure = _BodyFailed("body failed after the manifest was published")
    seen: list[tuple[object, object, object]] = []

    def abort(ref, exposure_ref, measure_ref) -> None:
        seen.append((ref, exposure_ref, measure_ref))
        raise failure

    catalog = _scratch_catalog(con, dialect)
    schema = f"ud_probe_{uuid4().hex[:12]}"
    _create_scratch_namespace(con, dialect, schema)
    try:
        store = WarehouseArtifactStore(con, schema_name=schema)
        with pytest.raises(_BodyFailed) as raised:
            _publish_then_abort(store, abort)
        assert raised.value is failure
        ((ref, exposure_ref, measure_ref),) = seen
        assert store.visible_manifests == ()

        fresh = WarehouseArtifactStore(con, schema_name=schema)
        assert fresh.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with fresh.open_snapshot(ref):
                pass
        assert refusal.value.code == "artifact.generation.dropped"
        remaining = set(con.list_tables(database=schema))
        assert {exposure_ref.relation.name, measure_ref.relation.name}.isdisjoint(remaining)

        # abandon_generation hides a pair whether or not a manifest row exists yet, and
        # repeating it (including on an already-invalidated pair) changes nothing.
        unseen = (uuid4(), uuid4())
        for pair in (unseen, unseen, (ref.artifact_id, ref.generation_id)):
            fresh.abandon_generation(*pair)
        tombstones = con.table("ud_manifest_dropped", database=schema).execute()
        hidden = set(zip(tombstones["artifact_id"], tombstones["generation_id"], strict=True))
        assert (str(unseen[0]), str(unseen[1])) in hidden
        assert len(tombstones) == len(hidden)
        assert fresh.visible_manifests == ()
    finally:
        _drop_scratch_namespace(con, schema, catalog)


def run_artifact_explicit_catalog_probe(con, dialect: str) -> None:
    """The explicit catalog-plus-schema store path works end to end on the live backend.

    A probe-owned scratch namespace is addressed as ``catalog=<scratch catalog>,
    schema_name=<scratch schema>`` (the catalog is the connection's current project
    or database on Snowflake and BigQuery). Snowflake qualifies ``create_table``
    with a dotted namespace; BigQuery uses a fully qualified table name and bare
    dataset. A minimal generation is published and left published; a
    FRESH store built the same way lists it, ``open_snapshot`` reads its manifest,
    ``drop_generation`` makes it invisible and refuses reads with
    ``artifact.generation.dropped``, and both relations are gone from the namespace.

    PostgreSQL has no scratch catalog, so there the probe degenerates to the
    schema-only path.
    """
    from uuid import uuid4

    import pytest

    from increment.query.artifact_contract import ArtifactContractError

    seen: list[tuple[object, object, object]] = []

    def keep(ref, exposure_ref, measure_ref) -> None:
        seen.append((ref, exposure_ref, measure_ref))

    catalog = _scratch_catalog(con, dialect)
    schema = f"ud_probe_{uuid4().hex[:12]}"
    _create_scratch_namespace(con, dialect, schema)
    try:
        store = WarehouseArtifactStore(con, catalog=catalog, schema_name=schema)
        _publish_then_abort(store, keep)
        ((ref, exposure_ref, measure_ref),) = seen

        fresh = WarehouseArtifactStore(con, catalog=catalog, schema_name=schema)
        assert [visible.manifest_sha256 for visible in fresh.visible_manifests] == [
            ref.manifest_sha256
        ]
        with fresh.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        assert manifest.artifact_id == ref.artifact_id
        assert manifest.generation_id == ref.generation_id
        relations = {exposure_ref.relation.name, measure_ref.relation.name}
        assert relations <= set(con.list_tables(database=schema))

        fresh.drop_generation(ref.artifact_id, ref.generation_id)
        assert fresh.visible_manifests == ()
        reread = WarehouseArtifactStore(con, catalog=catalog, schema_name=schema)
        assert reread.visible_manifests == ()
        with pytest.raises(ArtifactContractError) as refusal:
            with reread.open_snapshot(ref):
                pass
        assert refusal.value.code == "artifact.generation.dropped"
        assert relations.isdisjoint(set(con.list_tables(database=schema)))
    finally:
        _drop_scratch_namespace(con, schema, catalog)
