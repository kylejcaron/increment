"""Observable snapshot and persisted-row regressions, using in-memory DuckDB."""

import pytest

pytestmark = pytest.mark.slow


def test_pinned_streams_preserve_calendar_numeric_structured_and_empty_values():
    from datetime import UTC, date, datetime
    from decimal import Decimal

    import ibis

    from increment.errors import CapabilityError
    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    query = """SELECT DATE '2024-01-02' AS calendar,
        TIMESTAMP '2024-01-02 03:04:05' AS naive,
        TIMESTAMPTZ '2024-01-02 03:04:05+00' AS aware,
        9007199254740993::BIGINT AS integer,
        1.0000000000000002::DOUBLE AS neighboring,
        1.2345::DECIMAL(18,4) AS decimal_value,
        {'date': DATE '2024-01-02', 'offset': 2} AS structured,
        [1, NULL, 3] AS sequence,
        '2024-01-02' AS text_day, NULL::DOUBLE AS missing"""
    empty = "SELECT DATE '2024-01-01' AS ds, 2::BIGINT AS n WHERE FALSE"
    con = ibis.duckdb.connect()
    try:
        pinned = WarehouseSession(con, Definitions(dialect="duckdb")).pin_sources((query, empty))
        [row] = con.to_pyarrow(pinned.source_sql(query)).to_pylist()
        assert row == {
            "calendar": date(2024, 1, 2),
            "naive": datetime(2024, 1, 2, 3, 4, 5),
            "aware": datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
            "integer": 9007199254740993,
            "neighboring": 1.0000000000000002,
            "decimal_value": Decimal("1.2345"),
            "structured": {"date": date(2024, 1, 2), "offset": 2},
            "sequence": [1, None, 3],
            "text_day": "2024-01-02",
            "missing": None,
        }
        empty_rows = con.to_pyarrow(pinned.source_sql(empty))
        assert empty_rows.num_rows == 0
        assert empty_rows.column_names == ["ds", "n"]
        with pytest.raises(CapabilityError) as caught:
            pinned.source_sql("SELECT 3")
        assert caught.value.code == "query.session.snapshot.source_missing"
    finally:
        con.disconnect()


def test_artifact_write_persists_exactly_the_rows_it_digested():
    import ibis

    from increment.query.artifact_digest import context_sha256, digest_relation_sql_v2
    from increment.query.schemas import ARTIFACT_RELATION_PRIMARY_KEYS, ARTIFACT_RELATION_SCHEMAS
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.artifact import ArtifactContext

    payload = (
        '{"context_format":2,"experiment_name":"e",'
        '"window_days":{"end":null,"observation_horizon":null,"start":"2025-01-01"}}'
    )
    context = ArtifactContext(
        canonical_json=payload, sha256=context_sha256({"canonical_json": payload})
    )
    con = ibis.duckdb.connect()
    try:
        con.raw_sql("CREATE SEQUENCE revision START 1")
        store = WarehouseArtifactStore(con, schema_name="artifacts")
        live = con.sql("""SELECT 'e' AS experiment_id,
            CAST(nextval('revision') AS VARCHAR) AS unit_id, 'C' AS group_id,
            TIMESTAMPTZ '2024-01-01 00:00:00+00' AS first_exposure_ts,
            DATE '2024-01-01' AS first_exposure_date""")
        with store.begin_publication(expected_context=context) as publication:
            reference = publication.write_relation("exposures", live)
            assert reference.digest_format == 2
            persisted_table = con.table(reference.relation.name, database="artifacts")
            persisted = con.to_pyarrow(persisted_table)
            assert persisted["unit_id"].to_pylist() == ["1"]
            # Digest the already-persisted table directly (not a re-run of `live`,
            # whose `nextval` would advance): this is the same guarantee format-1's
            # `digest_relation` used to check, now against the SQL fast path.
            digest = digest_relation_sql_v2(
                con,
                persisted_table,
                "exposures",
                ARTIFACT_RELATION_SCHEMAS["exposures"],
                primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["exposures"],
            )
            assert digest.content_sha256 == reference.content_sha256
            assert digest.row_count == reference.row_count == 1
    finally:
        con.disconnect()


def test_pin_sources_is_one_server_side_snapshot_without_client_collection(monkeypatch):
    import ibis
    import ibis.expr.types as ir

    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    try:
        con.raw_sql("CREATE TABLE raw_left AS SELECT 1 AS revision, 10 AS value")
        con.raw_sql("CREATE TABLE raw_right AS SELECT 1 AS revision, 20 AS value")
        queries = ("SELECT * FROM raw_left", "SELECT * FROM raw_right")
        create_table = con.create_table
        snapshots = []

        def capture(name, obj, **kwargs):
            assert isinstance(obj, ir.Table)
            assert kwargs["temp"] is True
            snapshots.append(name)
            return create_table(name, obj, **kwargs)

        def no_collect(*args, **kwargs):
            pytest.fail("pin_sources collected raw relations into the client")

        with monkeypatch.context() as patch:
            patch.setattr(con, "create_table", capture)
            patch.setattr(con, "to_pyarrow", no_collect)
            pinned = WarehouseSession(con, Definitions(dialect="duckdb")).pin_sources(queries)
        assert len(snapshots) == 1
        con.raw_sql("UPDATE raw_left SET revision=2, value=99")
        con.raw_sql("UPDATE raw_right SET revision=2, value=99")
        for sql, value in zip(queries, (10, 20), strict=True):
            relation = pinned.source_sql(sql)
            assert snapshots[0] in str(ibis.to_sql(relation))
            assert con.to_pyarrow(relation).to_pylist() == [{"revision": 1, "value": value}]
            assert con.execute(relation.value.sum()) == value
        pinned.drop_materialized()
        assert snapshots[0] not in con.list_tables(database=("temp", "main"))
    finally:
        con.disconnect()


def test_pin_sources_never_falls_back_when_temp_creation_fails(monkeypatch):
    import ibis

    from increment.errors import CapabilityError
    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    try:

        def denied(*args, **kwargs):
            raise PermissionError("no TEMP privilege")

        monkeypatch.setattr(con, "create_table", denied)
        with pytest.raises(CapabilityError) as error:
            WarehouseSession(con, Definitions(dialect="duckdb")).pin_sources(("SELECT 1 AS x",))
        assert error.value.code == "query.session.snapshot.materialization_failed"
    finally:
        con.disconnect()


def test_pin_sources_scopes_a_fact_branch_to_enrolled_units_and_window():
    from datetime import datetime

    import ibis

    from increment.query.session import SourceScope, WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    try:
        con.raw_sql(
            """
            create table events as
            select * from (values
              ('u1', DATE '2024-01-05', 10.0),
              ('u2', DATE '2024-01-06', 20.0),
              ('u3', DATE '2023-06-01', 30.0)   -- outside the window
            ) t(user_id, event_at, value)
            """
        )
        con.raw_sql("create table enrolled as select * from (values ('u1'), ('u2')) t(unit_id)")
        session = WarehouseSession(con, Definitions(dialect="duckdb"))
        enrolled_units = con.table("enrolled").mutate(experiment_id=ibis.literal("exp1"))
        scope = SourceScope(
            enrolled_units=enrolled_units,
            window_start=datetime(2024, 1, 1),
        )
        pinned = session.pin_sources(
            ("SELECT * FROM events",),
            column_hints={"SELECT * FROM events": ("user_id", "event_at")},
            scope=scope,
        )
        result = con.to_pyarrow(pinned.source_sql("SELECT * FROM events")).to_pylist()
        assert {row["user_id"] for row in result} == {"u1", "u2"}
    finally:
        con.disconnect()


def test_pin_sources_without_scope_is_unchanged():
    import ibis

    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    try:
        con.raw_sql("create table t1 as select 1 as x")
        session = WarehouseSession(con, Definitions(dialect="duckdb"))
        pinned = session.pin_sources(("SELECT * FROM t1",))
        assert con.to_pyarrow(pinned.source_sql("SELECT * FROM t1")).to_pylist() == [{"x": 1}]
    finally:
        con.disconnect()


def test_pinned_source_window_retains_pre_period_and_last_day_events(tmp_path):
    """Regression: pin_sources's window scope must not drop CUPED
    pre-period events (before experiment.start) or events on the whole last
    observation day (models.py's day-granularity end semantics). Uses two
    separate fact sources -- one for the exposure, one for the metric -- so
    the metric's fact source is actually window-scoped (a fact source shared
    between exposure and metric is forced unscoped, per prior fix (c)).

    The metric's 14-day window closes after the declared end, so the fact
    source has to carry an event past that close (2025-01-15) for the
    enrolled unit to be observable at all; the readout then has to match the
    live source exactly, including the pre-period event."""
    from datetime import datetime

    import ibis
    import pyarrow as pa

    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "signups",
            obj=pa.table(
                {
                    "user_id": ["u1"],
                    "ts": [datetime(2025, 1, 1, 0, 0)],
                    "group_id": ["treatment"],
                }
            ),
        )
        con.create_table(
            "purchases",
            obj=pa.table(
                {
                    "user_id": ["u1", "u1", "u1", "u1"],
                    "ts": [
                        datetime(
                            2024, 12, 30, 12, 0
                        ),  # 2 days before start: within n_pre_periods=2
                        datetime(2025, 1, 5, 10, 0),  # ordinary in-window event
                        datetime(2025, 1, 9, 23, 30),  # in-window event at a late time-of-day
                        datetime(2025, 1, 15, 12, 0),  # past the window close: freshness tail
                    ],
                    "revenue": [1.0, 2.0, 3.0, 7.0],
                }
            ),
        )
        definitions_path = tmp_path / "pre_period.yml"
        definitions_path.write_text(
            """
dialect: duckdb
fact_sources:
  - name: signup_events
    sql: SELECT *, 'signup' AS event FROM signups
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: signup
        column: null
  - name: purchase_events
    sql: SELECT *, 'purchase' AS event FROM purchases
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: revenue
exposures:
  - name: assignment
    fact: signup
metrics:
  - name: rev
    type: mean
    entity: user_id
    fact: purchase
    window_days: 14
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-01-15T00:00:00
    n_pre_periods: 2
    control_group: control
    plan:
      primary: [rev]
"""
        )
        definitions = load(definitions_path)
        experiment = definitions.experiments[0]
        design = Randomized(control_group="control")
        native = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store="none",
            on_mixed_assignment="error",
            design=design,
            plan=compile_decision_plan(
                experiment.plan, definitions.metrics, path="warehouse", design=design
            ),
        )
        metric = definitions.metrics[0]
        live = native.moments(metric, include_covariate=True)
        with native.readout_snapshot(metrics=[metric], population="assigned") as pinned:
            snapshot = pinned.moments(metric, include_covariate=True)
        assert live, "the freshness-tail row must make the enrolled unit observable"
        assert snapshot == live
        assert [row["ref_x"] for row in snapshot] == [1.0]
        # ref_x counts the 2024-12-30 pre-period event; ref_y counts the
        # 2025-01-05 and 2025-01-09 in-window events (the metric aggregates
        # by count, so the 2025-01-15 freshness tail is outside the window).
        assert [row["ref_y"] for row in snapshot] == [2.0]
    finally:
        con.disconnect()


def test_pinned_source_window_does_not_clip_freshness_watermark(tmp_path):
    """Regression: pin_sources's window scope must not drop an event dated
    well past `observation_horizon` -- that is exactly how a fact source
    signals it has loaded past every unit's analysis window (the
    freshness/`data_as_of` watermark; see simulate/dgp.py's freshness-tail
    convention, which dates such events ~40 days past `experiment.start`
    specifically so no metric's censoring trips). Bounding the pinned
    snapshot's upper timestamp at `observation_horizon + 2 days` clips that
    signal, so `data_as_of` collapses back to the last in-window event and
    every enrolled unit is wrongly censored as unobservable."""
    from datetime import datetime

    import ibis
    import pyarrow as pa

    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "signups",
            obj=pa.table(
                {
                    "user_id": ["u1"],
                    "ts": [datetime(2025, 1, 1, 0, 0)],
                    "group_id": ["treatment"],
                }
            ),
        )
        con.create_table(
            "purchases",
            obj=pa.table(
                {
                    "user_id": ["u1", "u1"],
                    "ts": [
                        datetime(2025, 1, 5, 10, 0),  # ordinary in-window event
                        datetime(2025, 3, 1, 0, 0),  # far past the window: freshness tail
                    ],
                    "revenue": [2.0, 3.0],
                }
            ),
        )
        definitions_path = tmp_path / "freshness.yml"
        definitions_path.write_text(
            """
dialect: duckdb
fact_sources:
  - name: signup_events
    sql: SELECT *, 'signup' AS event FROM signups
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: signup
        column: null
  - name: purchase_events
    sql: SELECT *, 'purchase' AS event FROM purchases
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: revenue
exposures:
  - name: assignment
    fact: signup
metrics:
  - name: rev
    type: mean
    entity: user_id
    fact: purchase
    window_days: 14
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-02-01T00:00:00
    n_pre_periods: 0
    control_group: control
    plan:
      primary: [rev]
"""
        )
        definitions = load(definitions_path)
        experiment = definitions.experiments[0]
        design = Randomized(control_group="control")
        native = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store="none",
            on_mixed_assignment="error",
            design=design,
            plan=compile_decision_plan(
                experiment.plan, definitions.metrics, path="warehouse", design=design
            ),
        )
        metric = definitions.metrics[0]
        live = native.moments(metric)
        with native.readout_snapshot(metrics=[metric], population="assigned") as pinned:
            snapshot = pinned.moments(metric)
        assert [row["ref_y"] for row in live] == [1.0]
        assert snapshot == live, (
            "the freshness-tail event, dated well past observation_horizon, "
            "must survive pin_sources's window scope so data_as_of reflects "
            "the fact source's true watermark, not a window-clipped one"
        )
    finally:
        con.disconnect()


def test_pinned_source_freshness_watermark_scans_non_enrolled_units_too(tmp_path):
    """Regression: `_pinned_source` must not semi-join a metric fact source
    to enrolled units before `_data_as_of` reads it. A pipeline commonly
    lands late data for units outside any one experiment -- that is exactly
    the signal freshness needs -- so scoping this source by enrollment
    silently reverts the watermark to the last enrolled-unit event and
    wrongly censors every unit as unobservable."""
    from datetime import datetime

    import ibis
    import pyarrow as pa

    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "signups",
            obj=pa.table(
                {
                    "user_id": ["u1"],
                    "ts": [datetime(2025, 1, 1, 0, 0)],
                    "group_id": ["treatment"],
                }
            ),
        )
        con.create_table(
            "purchases",
            obj=pa.table(
                {
                    "user_id": ["u1", "zzz_never_enrolled"],
                    "ts": [datetime(2025, 1, 5, 10, 0), datetime(2025, 3, 1, 0, 0)],
                    "revenue": [2.0, 3.0],
                }
            ),
        )
        definitions_path = tmp_path / "freshness_unit_scope.yml"
        definitions_path.write_text(
            """
dialect: duckdb
fact_sources:
  - name: signup_events
    sql: SELECT *, 'signup' AS event FROM signups
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: signup
        column: null
  - name: purchase_events
    sql: SELECT *, 'purchase' AS event FROM purchases
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: revenue
exposures:
  - name: assignment
    fact: signup
metrics:
  - name: rev
    type: mean
    entity: user_id
    fact: purchase
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    n_pre_periods: 0
    control_group: control
    plan:
      primary: [rev]
"""
        )
        definitions = load(definitions_path)
        experiment = definitions.experiments[0]
        design = Randomized(control_group="control")
        metric = definitions.metrics[0]
        native = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store="none",
            on_mixed_assignment="error",
            design=design,
            plan=compile_decision_plan(
                experiment.plan, definitions.metrics, path="warehouse", design=design
            ),
        )
        purchase_fs = definitions.fact_sources[1]
        fact = purchase_fs.facts[0].name
        unpinned_watermark = con.execute(
            native._data_as_of(native._get_fact_table(purchase_fs), fact).as_table()
        )
        with native._pinned_source_execution(metrics=[metric]) as pinned:
            pinned_watermark = con.execute(
                pinned._data_as_of(pinned._get_fact_table(purchase_fs), fact).as_table()
            )
        assert pinned_watermark.iloc[0, 0] == unpinned_watermark.iloc[0, 0] == datetime(2025, 3, 1)
    finally:
        con.disconnect()


def test_pinned_source_retains_pre_window_breakout_property_rows(tmp_path):
    """Regression: `_pinned_source` must not push the window's lower time
    bound onto a source read only for a breakout/factor/covariate property
    lookup. `_build_breakout_properties_table` takes the latest value over
    the whole source (static) or strictly before exposure (pre_exposure),
    so a property row written earlier than window_start -- a signup-time
    tier, say -- must survive the pinned snapshot exactly as it does
    unpinned."""
    from datetime import datetime

    import ibis
    import pyarrow as pa

    from increment.analysis import Analysis
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "signups",
            obj=pa.table(
                {"user_id": ["u1"], "ts": [datetime(2025, 1, 1, 0, 0)], "group_id": ["treatment"]}
            ),
        )
        con.create_table(
            "purchases",
            obj=pa.table(
                {
                    "user_id": ["u1", "u1"],
                    "ts": [
                        datetime(2025, 1, 5, 10, 0),  # ordinary in-window event
                        datetime(2025, 1, 15, 12, 0),  # past the window close: freshness tail
                    ],
                    "revenue": [2.0, 7.0],
                }
            ),
        )
        con.create_table(
            "tiers",
            obj=pa.table(
                {
                    "user_id": ["u1"],
                    "ts": [datetime(2024, 12, 30, 12, 0)],  # signup-time tier, before window_start
                    "tier": ["gold"],
                }
            ),
        )
        definitions_path = tmp_path / "breakout_pre_window.yml"
        definitions_path.write_text(
            """
dialect: duckdb
fact_sources:
  - name: signup_events
    sql: SELECT *, 'signup' AS event FROM signups
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: signup
        column: null
  - name: purchase_events
    sql: SELECT *, 'purchase' AS event FROM purchases
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: revenue
  - name: tier_events
    sql: SELECT *, 'tier_set' AS event FROM tiers
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: tier_set
        column: null
    properties:
      - name: tier
        column: tier
        dtype: string
        as_of: pre_exposure
exposures:
  - name: assignment
    fact: signup
metrics:
  - name: rev
    type: mean
    entity: user_id
    fact: purchase
    window_days: 14
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    n_pre_periods: 0
    control_group: control
    breakouts:
      - property: tier
        source: tier_events
        skip_missing: true
    plan:
      primary: [rev]
"""
        )
        definitions = load(definitions_path)
        native = Analysis.from_definitions("exp", definitions_path, con)
        context = artifact_context(definitions, native.experiment, "error")
        selected = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind in ("breakout_dimension", "assignment_counts")
        ]
        store = WarehouseArtifactStore(con, schema_name="artifacts")
        ref = native.publish_unit_day_artifact(store, extensions=selected)
        adopted = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
        try:
            summaries = adopted.breakout_summaries(metrics=["rev"])
            tiers = {
                row["tier"]
                for row in summaries["rev:tier:tier_events"]["group_summary"].to_pylist()
            }
            assert tiers == {"gold"}, (
                "a breakout property row from before window_start must survive "
                "the pinned snapshot, matching what the unpinned path reads"
            )
        finally:
            adopted.close()
    finally:
        con.disconnect()


def test_readout_snapshot_pins_declared_observational_covariate_source(tmp_path):
    """Regression: `readout_snapshot` must pin an observational design's
    declared covariate source, not only metric and uptake sources. Without
    it, `unit_frame(..., covariates=[...])` against the pinned source raises
    `query.session.snapshot.source_missing` even though the covariate is
    declared under `design.covariates`."""
    from datetime import datetime

    import ibis
    import narwhals as nw
    import pyarrow as pa

    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.loader import load

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "exposures",
            obj=pa.table(
                {
                    "user_id": ["u1", "u2"],
                    "ts": [datetime(2025, 1, 1, 0, 0)] * 2,
                    "group_id": ["treatment", "control"],
                }
            ),
        )
        con.create_table(
            "purchases",
            obj=pa.table(
                {
                    "user_id": ["u1", "u2", "u1"],
                    "ts": [
                        datetime(2025, 1, 2, 0, 0),
                        datetime(2025, 1, 2, 0, 0),
                        datetime(2025, 2, 1, 0, 0),  # freshness anchor: proves the window closed
                    ],
                    "revenue": [3.0, 4.0, 1.0],
                }
            ),
        )
        con.create_table(
            "users",
            obj=pa.table(
                {
                    "user_id": ["u1", "u2"],
                    "updated_at": [datetime(2024, 12, 1, 0, 0)] * 2,
                    "tenure_days": [30.0, 45.0],
                }
            ),
        )
        definitions_path = tmp_path / "observational_covariate.yml"
        definitions_path.write_text(
            """
dialect: duckdb
fact_sources:
  - name: exposure_events
    sql: SELECT *, 'saw_feature' AS event FROM exposures
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: saw_feature
        column: null
  - name: purchase_events
    sql: SELECT *, 'purchase' AS event FROM purchases
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: revenue
  - name: users
    sql: SELECT * FROM users
    timestamp_column: updated_at
    entities: [user_id]
    facts:
      - name: user_update
        column: null
    properties:
      - name: tenure_days
        column: tenure_days
        dtype: float
        as_of: pre_exposure
exposures:
  - name: assignment
    fact: saw_feature
metrics:
  - name: rev
    type: mean
    entity: user_id
    fact: purchase
    window_days: 14
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    n_pre_periods: 0
    control_group: control
    design:
      mechanism: observational
      covariates:
        - {property: tenure_days, source: users}
    plan:
      primary: [rev]
"""
        )
        definitions = load(definitions_path)
        experiment = definitions.experiments[0]
        metric = definitions.metrics[0]
        native = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store="none",
            on_mixed_assignment="error",
            design=experiment.resolved_design(),
            plan=compile_decision_plan(
                experiment.plan,
                definitions.metrics,
                path="warehouse",
                design=experiment.resolved_design(),
            ),
        )
        unpinned = nw.from_native(native.unit_frame(metric, covariates=["tenure_days"]))
        with native.readout_snapshot(metrics=[metric], population="assigned") as pinned:
            pinned_frame = nw.from_native(pinned.unit_frame(metric, covariates=["tenure_days"]))
        assert pinned_frame.columns == unpinned.columns
        assert set(pinned_frame.get_column("unit_id").to_list()) == set(
            unpinned.get_column("unit_id").to_list()
        )
    finally:
        con.disconnect()


@pytest.mark.parametrize("filtered", [True, False])
def test_pinned_freshness_keeps_non_enrolled_joined_dimension_rows(filtered):
    """Freshness must retain dimension rows needed by non-enrolled fact events."""
    from datetime import datetime

    import ibis

    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "events",
            obj=ibis.memtable(
                [
                    {
                        "user_id": "a",
                        "ts": datetime(2025, 1, 1, 12),
                        "event": "enrolled",
                        "group_id": "C",
                        "amount": 0.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": "b",
                        "ts": datetime(2025, 1, 1, 12),
                        "event": "enrolled",
                        "group_id": "T",
                        "amount": 0.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": "a",
                        "ts": datetime(2025, 1, 2, 12),
                        "event": "purchase",
                        "group_id": "C",
                        "amount": 10.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": "b",
                        "ts": datetime(2025, 1, 2, 12),
                        "event": "purchase",
                        "group_id": "T",
                        "amount": 20.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": "outside",
                        "ts": datetime(2025, 1, 10, 12),
                        "event": "purchase",
                        "group_id": "C",
                        "amount": 1.0,
                        "experiment_id": "exp",
                    },
                ]
            ),
        )
        con.create_table(
            "users",
            obj=ibis.memtable({"user_id": ["a", "b", "outside"], "country": ["UK", "UK", "UK"]}),
        )
        definitions = Definitions.model_validate(
            {
                "dialect": "duckdb",
                "dim_sources": [
                    {
                        "name": "users",
                        "sql": "SELECT * FROM users",
                        "entity": "user_id",
                        "properties": [{"name": "country", "column": "country", "as_of": "static"}],
                    }
                ],
                "fact_sources": [
                    {
                        "name": "events",
                        "sql": "SELECT * FROM events",
                        "timestamp_column": "ts",
                        "entities": ["user_id"],
                        "dims": ["users"],
                        "facts": [
                            {"name": "enrolled", "column": None},
                            {"name": "purchase", "column": "amount"},
                        ],
                    }
                ],
                "exposures": [{"name": "assignment", "fact": "enrolled"}],
                "metrics": [
                    {
                        "name": "revenue",
                        "type": "mean",
                        "entity": "user_id",
                        "fact": "purchase",
                        "aggregation": "sum",
                        "window_days": 7,
                        "filters": (
                            [{"property": "country", "op": "equals", "values": ["UK"]}]
                            if filtered
                            else []
                        ),
                    },
                ],
                "experiments": [
                    {
                        "name": "exp",
                        "exposure": "assignment",
                        "unit": "user_id",
                        "start": "2025-01-01T00:00:00",
                        "n_pre_periods": 0,
                        "control_group": "C",
                        "plan": {"primary": "revenue"},
                    }
                ],
            }
        )
        experiment = definitions.experiments[0]
        design = experiment.resolved_design()
        native = DefinitionsMomentSource(
            WarehouseSession(con, definitions),
            experiment,
            definitions.metrics,
            store="none",
            on_mixed_assignment="error",
            design=design,
            plan=compile_decision_plan(
                experiment.plan, definitions.metrics, path="warehouse", design=design
            ),
        )
        try:
            pinned = native._pinned_source()
            try:
                ordinary_horizon = con.execute(native._union_event_horizon())
                pinned_horizon = con.execute(pinned._union_event_horizon())
                assert ordinary_horizon == pinned_horizon == datetime(2025, 1, 10)

                for metric in definitions.metrics:
                    ordinary = sorted(
                        native.moments(metric),
                        key=lambda row: (row["metric"], row["group_id"]),
                    )
                    snapshot = sorted(
                        pinned.moments(metric),
                        key=lambda row: (row["metric"], row["group_id"]),
                    )
                    assert ordinary
                    assert snapshot == ordinary
                    assert {row["group_id"] for row in snapshot} == {"C", "T"}
                    assert sum(row["n"] for row in snapshot) == 2
            finally:
                pinned.close()
        finally:
            native.close()
    finally:
        con.disconnect()
