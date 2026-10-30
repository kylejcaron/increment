# integration/warehouse_execution/test_postgres_execution.py
"""Real PostgreSQL execution through the actual public Analysis entry
points: a ratio metric, materialized-cache correction, artifact
publish/adopt parity, and dataframe-oracle parity for CUPED and the
binary sequential routes -- not a hand-assembled query-builder pipeline."""

from __future__ import annotations

import datetime as dt
from contextlib import nullcontext
from uuid import uuid4

import pytest

from increment import Analysis
from increment.estimators import MetricTrends
from increment.query.session import WarehouseArtifactStore
from increment.semantics.models import Definitions
from integration.warehouse_execution import _suite
from integration.warehouse_execution._suite import (
    _postgres_temp_schema,
    run_artifact_abort_probe,
    run_artifact_confidentiality_probe,
    run_artifact_explicit_catalog_probe,
    run_artifact_namespace_probe,
    run_artifact_publish_adopt_probe,
    run_artifact_sequential_identity_probe,
    run_materialized_cache_correction_probe,
    run_ratio_metric_probe,
)
from tests._override_warnings import allow_declared_overrides

pytestmark = pytest.mark.warehouse_postgres


def test_ratio_metric_lift(postgres_con):
    run_ratio_metric_probe(postgres_con, dialect="postgres")


@pytest.fixture
def postgres_nonpublic_con(postgres_con):
    schema = f"cleanup_probe_{uuid4().hex}"
    postgres_con.create_database(schema)
    try:
        with postgres_con.raw_sql(f'SET search_path TO "{schema}"'):
            pass
        assert postgres_con.current_database == schema
        yield postgres_con
    finally:
        with postgres_con.raw_sql("RESET search_path"):
            pass
        with postgres_con.raw_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'):
            pass


def test_materialized_cache_reflects_true_correction(postgres_nonpublic_con, monkeypatch):
    run_materialized_cache_correction_probe(
        postgres_nonpublic_con, dialect="postgres", monkeypatch=monkeypatch
    )


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_materialized_cache_detects_skipped_cleanup(postgres_nonpublic_con, monkeypatch, cleanup):
    con = postgres_nonpublic_con
    drop_table = con.drop_table
    close = Analysis.close
    skipped = set()

    def skip_materialized_drop(name, *args, **kwargs):
        if name.startswith(("exp_spine_", "exp_stats_revenue_")):
            skipped.add(name)
            return None
        return drop_table(name, *args, **kwargs)

    def close_without_cleanup(analysis):
        with monkeypatch.context() as patch:
            patch.setattr(con, "drop_table", skip_materialized_drop)
            return close(analysis)

    if cleanup == "invalidation":
        monkeypatch.setattr(con, "drop_table", skip_materialized_drop)
    else:
        monkeypatch.setattr(Analysis, "close", close_without_cleanup)
    try:
        with pytest.raises(AssertionError, match="materialization leaked"):
            run_materialized_cache_correction_probe(
                con, dialect="postgres", monkeypatch=monkeypatch
            )
        assert skipped, "mutation did not intercept cleanup"
    finally:
        if skipped:
            schema = _postgres_temp_schema(con)
            for name in skipped:
                drop_table(name, database=schema, force=True)


@pytest.mark.parametrize(
    ("clustered", "fail_adoption"),
    [(True, False), (True, True), (False, False), (False, True)],
    ids=["clustered", "clustered-failed-adoption", "breakout", "breakout-failed-adoption"],
)
def test_artifact_publish_adopt_parity(
    postgres_nonpublic_con, monkeypatch, clustered, fail_adoption
):
    con = postgres_nonpublic_con
    WarehouseArtifactStore(con)
    metadata = ("ud_manifest_index", "ud_manifest_dropped")
    real_to_pyarrow = con.to_pyarrow
    forbidden_columns = (
        {"unit_id", "group_id", "first_exposure_ts", "first_exposure_date"},
        {"unit_id", "ds", "n_events", "sum_value", "min_value", "max_value"},
    )

    def reject_unit_transfer(table, *args, **kwargs):
        if hasattr(table, "schema"):
            names = set(table.schema().names)
            assert not any(columns <= names for columns in forbidden_columns), (
                "unit-level artifact rows crossed the Python materialization boundary"
            )
        return real_to_pyarrow(table, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", reject_unit_transfer)
    before = {name: con.table(name).to_pyarrow().to_pylist() for name in metadata}
    tables = {name for name in con.list_tables() if name.startswith("ud_")}

    def deny_namespace(*args, **kwargs):
        pytest.fail("artifact probe must use the configured namespace")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated adoption failure")

    monkeypatch.setattr(con, "create_database", deny_namespace)
    if fail_adoption:
        monkeypatch.setattr(Analysis, "from_unit_day_artifact", fail)
    expectation = (
        pytest.raises(RuntimeError, match="simulated adoption failure")
        if fail_adoption
        else nullcontext()
    )
    with expectation:
        run_artifact_publish_adopt_probe(con, dialect="postgres", clustered=clustered)
    assert {name for name in con.list_tables() if name.startswith("ud_")} == tables
    assert {name: con.table(name).to_pyarrow().to_pylist() for name in metadata} == before


def test_artifact_output_confidentiality(postgres_con):
    run_artifact_confidentiality_probe(postgres_con, dialect="postgres")


def test_postgres_sha256_hashes_utf8_without_decoding_backslashes(postgres_con):
    import hashlib

    import pyarrow as pa

    from increment.query.artifact_digest import (
        _sql_binary_hex_expr,
        _sql_digest_sha256_binary_expr,
        _sql_digest_sha256_expr,
    )

    values = ["", r"\x4142", r"a\b", r"a\000b", "é/多言語", "quote'\nline"]
    name = f"hash_utf8_spec_{uuid4().hex[:10]}"
    table = postgres_con.create_table(name, pa.table({"text": values}))
    try:
        binary = _sql_digest_sha256_binary_expr(table.text, dialect="postgres")
        actual = postgres_con.to_pyarrow(
            table.select(
                text=table.text,
                hex_hash=_sql_digest_sha256_expr(table.text, dialect="postgres"),
                binary_hash=_sql_binary_hex_expr(binary, dialect="postgres"),
            )
        ).to_pylist()
        assert len(actual) == len(values)
        for row in actual:
            expected = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            assert row["hex_hash"] == row["binary_hash"] == expected
    finally:
        postgres_con.drop_table(name, force=True)


def test_postgres_timestamp_tokens_preserve_epoch_microseconds(postgres_con):
    import pyarrow as pa

    from increment.query.artifact_digest import _SQL_DIGEST_TIMESTAMP_TOKEN

    instants = [
        dt.datetime(1, 1, 1, 0, 0, 0, 1, tzinfo=dt.UTC),
        dt.datetime(1900, 1, 1, 0, 0, 0, 1, tzinfo=dt.UTC),
        dt.datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=dt.UTC),
        dt.datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=dt.UTC),
        dt.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=dt.UTC),
    ]
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    expected = []
    for value in instants:
        delta = value - epoch
        expected.append(str((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds))
    name = f"timestamp_token_spec_{uuid4().hex[:10]}"
    table = postgres_con.create_table(
        name, pa.table({"ts": pa.array(instants, type=pa.timestamp("us", tz="UTC"))})
    )
    try:
        for zone in ("UTC", "America/Los_Angeles"):
            postgres_con.raw_sql(f"SET TIME ZONE '{zone}'")
            actual = (
                postgres_con.to_pyarrow(
                    table.select(token=_SQL_DIGEST_TIMESTAMP_TOKEN["postgres"](table.ts))
                )
                .column("token")
                .to_pylist()
            )
            assert sorted(actual) == sorted(expected)
    finally:
        postgres_con.raw_sql("SET TIME ZONE 'UTC'")
        postgres_con.drop_table(name, force=True)


def test_postgres_format4_digest_matches_duckdb(postgres_con):
    """Exercise format-4 hashing on native PostgreSQL values using its built-in SHA-256 path."""
    import ibis

    from increment.query.artifact_digest import digest_relation_sql_v2
    from tests.test_unit_day_artifact_digest_v2 import (
        _PORTABLE_SCHEMA,
        _portable_digest_arrow,
    )

    schema = _PORTABLE_SCHEMA
    arrow = _portable_digest_arrow()
    duckdb = ibis.duckdb.connect()
    try:
        duck_table = duckdb.create_table("digest_spec", arrow)
        pg_table = postgres_con.create_table(f"digest_spec_{uuid4().hex[:10]}", arrow)
        expected = digest_relation_sql_v2(
            duckdb, duck_table, "portable_spec", schema, primary_key=("id",)
        )
        actual = digest_relation_sql_v2(
            postgres_con, pg_table, "portable_spec", schema, primary_key=("id",)
        )
        assert (
            expected.content_sha256
            == "28b8c5135e6da8dcddeca6854a0bb9cc86a205eee4f6b3e26795589124e669e5"
        )
        from increment.query.artifact_digest import _row_token_expr

        fields = [
            ("id", 2, "int"),
            ("text", 1, "string"),
            ("integer", 2, "int"),
            ("float", 3, "float"),
            ("date", 4, "date"),
            ("timestamp", 5, "timestamp"),
            ("boolean", 6, "bool"),
        ]
        duck_tokens = duckdb.to_pyarrow(
            duck_table.select(
                id=duck_table.id, token=_row_token_expr(duck_table, fields, dialect="duckdb")
            )
        ).to_pylist()
        postgres_tokens = postgres_con.to_pyarrow(
            pg_table.select(
                id=pg_table.id,
                token=_row_token_expr(pg_table, fields, dialect="postgres"),
            )
        ).to_pylist()
        assert sorted(duck_tokens, key=lambda row: row["id"]) == sorted(
            postgres_tokens, key=lambda row: row["id"]
        )
        assert actual.content_sha256 == expected.content_sha256
    finally:
        duckdb.disconnect()
        if "pg_table" in locals():
            postgres_con.drop_table(pg_table.get_name(), force=True)


def test_postgres_format4_date_tokens_ignore_datestyle(postgres_con):
    import ibis

    from increment.query.artifact_digest import digest_relation_sql_v2

    table_name = f"date_style_digest_{uuid4().hex[:10]}"
    schema = (("id", "INT64", False), ("day", "DATE", False))
    relation = None
    original_datestyle = postgres_con.raw_sql("SHOW DateStyle").fetchone()[0]
    try:
        relation = postgres_con.create_table(
            table_name,
            ibis.memtable(
                [(1, dt.date(2024, 1, 2))],
                schema=ibis.schema({"id": "int64", "day": "date"}),
            ),
        )
        legacy_casts = []
        digests = []
        for datestyle in ("ISO, MDY", "SQL, MDY"):
            postgres_con.raw_sql(f"SET DateStyle TO '{datestyle}'")
            legacy_casts.append(
                postgres_con.to_pyarrow(relation.select(day_text=relation.day.cast("string")))[
                    "day_text"
                ].to_pylist()
            )
            digests.append(
                digest_relation_sql_v2(
                    postgres_con, relation, "date_style", schema, primary_key=("id",)
                ).content_sha256
            )
        assert legacy_casts[0] != legacy_casts[1]
        assert digests[0] == digests[1]
    finally:
        postgres_con.raw_sql(f"SET DateStyle TO '{original_datestyle}'")
        if relation is not None:
            postgres_con.drop_table(table_name, force=True)


def test_postgres_float64_token_matches_duckdb_random_binary64(postgres_con):
    from integration.warehouse_execution._float_tokens import run_float_token_probe

    run_float_token_probe(postgres_con, dialect="postgres", random_count=10_000)


@pytest.mark.parametrize("damage", ["none", "tamper", "duplicate", "domain", "exposure"])
def test_postgres_cuped_artifact_integrity_is_warehouse_only(
    postgres_con, monkeypatch, tmp_path, damage
):
    from integration.warehouse_execution._artifact_integrity import run_artifact_integrity_probe

    run_artifact_integrity_probe(
        postgres_con, dialect="postgres", monkeypatch=monkeypatch, tmp_path=tmp_path, damage=damage
    )


def test_artifact_sequential_identity_matches_native(postgres_con):
    run_artifact_sequential_identity_probe(postgres_con, dialect="postgres")


def test_artifact_manifest_index_namespace_contract(postgres_con):
    run_artifact_namespace_probe(postgres_con, dialect="postgres")


def test_conversion_calendar_trend(postgres_con):
    con = postgres_con
    name = "events_conversion_trend"
    definitions = Definitions.model_validate(
        {
            "dialect": "postgres",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": None},
                    ],
                }
            ],
            "metrics": [
                {
                    "name": "purchase_rate",
                    "type": "conversion",
                    "entity": "user_id",
                    "fact": "purchase",
                }
            ],
        }
    )
    rows = [
        {"user_id": user, "ts": dt.datetime(2025, 1, day, 12), "event": event}
        for user, day, event in [
            ("u1", 1, "purchase"),
            ("u1", 1, "purchase"),
            ("u2", 1, "page_view"),
            ("u1", 2, "page_view"),
            ("u2", 2, "page_view"),
            ("u1", 3, "purchase"),
            ("u2", 3, "purchase"),
        ]
    ]
    con.create_table(name, obj=rows)
    try:
        frame = (
            MetricTrends.from_definitions(definitions, con)
            .metric(
                "purchase_rate", grain="day", start=dt.date(2025, 1, 1), end=dt.date(2025, 1, 3)
            )
            .to_frame()
        )
        frame = frame.sort_values("period")
        assert frame["n"].tolist() == [2, 2, 2]
        assert frame["value"].tolist() == pytest.approx([0.5, 0.0, 1.0])
        assert frame["ci_lb"].notna().all()
        assert frame["ci_ub"].notna().all()
    finally:
        con.drop_table(name, force=True)


def test_denied_temp_creation_fails_the_gate(postgres_nonpublic_con, monkeypatch):
    """A denied TEMP creation refuses the default captured source snapshot."""
    from increment.errors import CodedError

    con = postgres_nonpublic_con
    create_table = con.create_table

    def deny_temp(name, *args, **kwargs):
        if kwargs.get("temp") is True:
            raise RuntimeError("simulated: permission denied for CREATE TEMP TABLE")
        return create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", deny_temp)
    with pytest.raises(CodedError) as refusal:
        run_materialized_cache_correction_probe(
            con, dialect="postgres", monkeypatch=monkeypatch, coherence="captured"
        )
    assert refusal.value.code == "query.session.snapshot.materialization_failed"


def test_partial_materialization_fails_the_gate(postgres_nonpublic_con, monkeypatch):
    """``store='always'`` must materialize BOTH the spine and
    revenue-stats TEMP tables on every readout. If only the spine lands
    (e.g. the backend denies just the stats create mid-flight), the
    gate must catch the partial materialization rather than treat spine
    alone as sufficient."""
    con = postgres_nonpublic_con
    create_table = con.create_table

    def deny_stats_temp(name, *args, **kwargs):
        if kwargs.get("temp") is True and name.startswith("exp_stats_revenue_"):
            raise RuntimeError("simulated: revenue-stats materialization denied")
        return create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", deny_stats_temp)
    with pytest.raises((AssertionError, UserWarning), match="revenue stats|could not materialize"):
        run_materialized_cache_correction_probe(con, dialect="postgres", monkeypatch=monkeypatch)


def test_duplicate_spine_creation_without_stats_fails_the_gate(postgres_nonpublic_con, monkeypatch):
    """Two physical spine materializations cannot substitute for a
    missing revenue-stats materialization -- the two required roles are
    checked independently by name prefix, never satisfied by volume."""
    con = postgres_nonpublic_con
    create_table = con.create_table
    duplicates: list[str] = []

    def duplicate_spine_deny_stats(name, *args, **kwargs):
        if kwargs.get("temp") is True and name.startswith("exp_stats_revenue_"):
            raise RuntimeError("simulated: revenue-stats materialization denied")
        relation = create_table(name, *args, **kwargs)
        if kwargs.get("temp") is True and name.startswith("exp_spine_"):
            dup_name = f"{name}_dup"
            create_table(dup_name, *args, **kwargs)
            duplicates.append(dup_name)
        return relation

    monkeypatch.setattr(con, "create_table", duplicate_spine_deny_stats)
    try:
        with pytest.raises(
            (AssertionError, UserWarning), match="revenue stats|could not materialize"
        ):
            run_materialized_cache_correction_probe(
                con, dialect="postgres", monkeypatch=monkeypatch
            )
    finally:
        for name in duplicates:
            con.drop_table(name, force=True)


def test_extra_unrelated_temp_creation_does_not_fail_the_gate(postgres_nonpublic_con, monkeypatch):
    """An extra, unrelated TEMP table created alongside the required
    spine/revenue-stats materializations must not trip the required
    -role check: only the two named roles are required, extra
    relations are tolerated."""
    con = postgres_nonpublic_con
    create_table = con.create_table
    decoys: list[str] = []

    def create_with_decoy(name, *args, **kwargs):
        relation = create_table(name, *args, **kwargs)
        if kwargs.get("temp") is True and name.startswith("exp_spine_"):
            decoy = f"{name}_unrelated_decoy"
            create_table(decoy, *args, **kwargs)
            decoys.append(decoy)
        return relation

    monkeypatch.setattr(con, "create_table", create_with_decoy)
    try:
        run_materialized_cache_correction_probe(con, dialect="postgres", monkeypatch=monkeypatch)
    finally:
        for name in decoys:
            con.drop_table(name, force=True)


def _run_permanent_shadow_probe(con, monkeypatch, cleanup, *, bare_cleanup=False):
    """Keep permanent rows intact when the owned TEMP relation is already absent."""
    schema = con.current_database
    drop_table = con.drop_table
    required_materializations = _suite._required_materializations
    sentinel_names = set()
    readout = 0
    target_readout = 1 if cleanup == "invalidation" else 2

    def remove_owned_relations(connection, created):
        nonlocal readout
        required = required_materializations(connection, created)
        readout += 1
        if readout == target_readout:
            for relation in required:
                name = relation.op().name
                sentinel_names.add(name)
                con.create_table(name, obj=[{"sentinel": 314159}], database=schema)
                assert con.table(name, database=schema).to_pyarrow().to_pylist() == [
                    {"sentinel": 314159}
                ]
                drop_table(name, database=_suite._materialization_database(relation), force=True)
        return required

    def bare_name_drop(name, *args, **kwargs):
        if name.startswith(("exp_spine_", "exp_stats_revenue_")):
            return drop_table(name, force=True)
        return drop_table(name, *args, **kwargs)

    monkeypatch.setattr(_suite, "_required_materializations", remove_owned_relations)
    if bare_cleanup:
        monkeypatch.setattr(con, "drop_table", bare_name_drop)
    try:
        run_materialized_cache_correction_probe(con, dialect="postgres", monkeypatch=monkeypatch)
        for name in sentinel_names:
            assert name in con.list_tables(database=schema), (
                f"permanent shadow sentinel {name!r} was deleted"
            )
            rows = con.table(name, database=schema).to_pyarrow().to_pylist()
            assert rows == [{"sentinel": 314159}], (
                f"permanent shadow sentinel {name!r} changed: {rows}"
            )
    finally:
        for name in sentinel_names:
            drop_table(name, database=schema, force=True)


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_permanent_shadow_sentinel_survives_cleanup(postgres_nonpublic_con, monkeypatch, cleanup):
    _run_permanent_shadow_probe(postgres_nonpublic_con, monkeypatch, cleanup)


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_permanent_shadow_sentinel_bare_name_cleanup_mutation(
    postgres_nonpublic_con, monkeypatch, cleanup
):
    with pytest.raises(AssertionError, match="permanent shadow sentinel"):
        _run_permanent_shadow_probe(postgres_nonpublic_con, monkeypatch, cleanup, bare_cleanup=True)


@allow_declared_overrides
def test_warehouse_cuped_matches_the_dataframe_oracle(postgres_con, tmp_path):
    """Ratio CUPED and the adjusted sequential laws through both warehouse
    readers on live PostgreSQL agree with the dataframe oracle."""
    from tests.warehouse_cuped_cases import (
        check_day_axis_parity,
        check_fixed_horizon_parity,
        check_sequential_parity,
        event_rows,
        unit_rows,
    )

    source = _suite._qualified_table_identifier(postgres_con, "postgres", "events_cuped_parity")
    postgres_con.create_table("events_cuped_parity", obj=event_rows(unit_rows()), overwrite=True)
    try:
        check_fixed_horizon_parity(postgres_con, "postgres", source, tmp_path)
        check_sequential_parity(postgres_con, "postgres", source)
        check_day_axis_parity(postgres_con, "postgres", source)
    finally:
        postgres_con.drop_table("events_cuped_parity", force=True)


@pytest.mark.parametrize("kind", ["asymptotic_mean", "always_valid"])
def test_binary_sequential_route_matches_the_dataframe_oracle(postgres_con, kind):
    """A conversion metric on each automatic binary route through both
    warehouse readers on live PostgreSQL retains the dataframe oracle's
    sequential state and interval byte for byte."""
    from tests.binary_sequential_cases import check_path_parity, event_rows, unit_rows

    # One table per parameter: the two invocations otherwise share a name on one
    # database, so a parallel run would drop a table the sibling is still reading.
    table = f"events_binary_parity_{kind}"
    source = _suite._qualified_table_identifier(postgres_con, "postgres", table)
    postgres_con.create_table(table, obj=event_rows(unit_rows()), overwrite=True)
    try:
        check_path_parity(postgres_con, "postgres", source, kind)
    finally:
        postgres_con.drop_table(table, force=True)


@pytest.fixture
def postgres_event_connection(postgres_connect):
    import pyarrow as pa

    def connection(rows):
        con = postgres_connect()
        try:
            table = pa.Table.from_pylist(rows)
            for i, field in enumerate(table.schema):
                if pa.types.is_null(field.type):
                    table = table.set_column(i, field.name, table.column(i).cast(pa.float64()))
            con.create_table("events", obj=table, temp=True)
        except BaseException:
            con.disconnect()
            raise
        return con

    return connection


@pytest.mark.parametrize("default_sensitivity", [False, True])
def test_three_arm_observational_methods_match_all_ingress_paths(
    postgres_event_connection, default_sensitivity
):
    from tests.parity_harness.cases import _observational_aipw_dml_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _observational_aipw_dml_case(
        warehouse_connection=postgres_event_connection,
        dialect="postgres",
        default_sensitivity=default_sensitivity,
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("missing_levels", [False, True], ids=["all-levels", "null-levels"])
def test_categorical_covariate_matches_all_ingress_paths(postgres_event_connection, missing_levels):
    """A declared string property adjusts as a categorical covariate through
    the live PostgreSQL warehouse read and the artifact it publishes, row
    for row with the dataframe oracle; a NULL level reaches both as a
    missing value and refuses by name under the definitions' default
    policy while the frame paths keep every unit under impute-indicator."""
    from tests.parity_harness.cases import _observational_categorical_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _observational_categorical_case(
        missing_levels=missing_levels,
        warehouse_connection=postgres_event_connection,
        dialect="postgres",
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("kind", ["normal", "student_t", "mixture"])
def test_lift_priors_match_all_ingress_paths(postgres_event_connection, kind):
    from dataclasses import replace

    from increment.estimation.engine import Method
    from tests.parity_harness.cases import _lift_prior_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _lift_prior_case(
        kind, warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    case = replace(case, decision_method=Method(name="unadjusted"))
    assert_parity(case, run_case(case))


def test_clearing_prior_matches_all_ingress_paths(postgres_event_connection):
    from dataclasses import replace

    from increment.estimation.engine import Method
    from tests.parity_harness.cases import _prior_reset_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _prior_reset_case(warehouse_connection=postgres_event_connection, dialect="postgres")
    case = replace(case, decision_method=Method(name="unadjusted"))
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("metric_free", [False, True], ids=["missing-outcomes", "empty-catalog"])
def test_compliance_only_missing_outcomes_match_all_ingress_paths(
    postgres_event_connection, metric_free
):
    from tests.parity_harness.cases import _encouragement_declared_definitions_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _encouragement_declared_definitions_case(
        missing_treatment_outcomes=not metric_free,
        metric_free=metric_free,
        warehouse_connection=postgres_event_connection,
        dialect="postgres",
    )
    assert_parity(case, run_case(case))


def test_metric_free_sequential_compliance_matches_all_ingress_paths(postgres_event_connection):
    from tests.parity_harness.cases import _sequential_composed_itt_and_uptake_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _sequential_composed_itt_and_uptake_case(
        metric_free=True, warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("sequential", [False, True], ids=["fixed", "sequential"])
def test_optional_exclusion_matches_all_ingress_paths(postgres_event_connection, sequential):
    from tests.parity_harness.cases import (
        _encouragement_declared_definitions_case,
        _sequential_composed_itt_and_uptake_case,
    )
    from tests.parity_harness.runner import assert_parity, run_case

    factory = (
        _sequential_composed_itt_and_uptake_case
        if sequential
        else _encouragement_declared_definitions_case
    )
    case = factory(
        exclusion_declared=False,
        warehouse_connection=postgres_event_connection,
        dialect="postgres",
    )
    assert_parity(case, run_case(case))


def test_sequential_missing_zero_matches_all_ingress_paths(postgres_event_connection):
    from tests.parity_harness.cases import _sequential_missing_zero_mean_ratio_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _sequential_missing_zero_mean_ratio_case(
        warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("complete", [False, True], ids=["lagging-denominator", "both-complete"])
def test_ratio_component_horizons_match_all_ingress_paths(postgres_event_connection, complete):
    from tests.parity_harness.cases import _ratio_component_horizon_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _ratio_component_horizon_case(
        complete=complete, warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case, coherence="live"))


def test_uptake_timestamp_boundaries_match_all_ingress_paths(postgres_event_connection):
    from tests.parity_harness.cases import _uptake_timestamp_boundaries_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _uptake_timestamp_boundaries_case(
        warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("outcome_day", [1, 3], ids=["unfinished", "mature"])
def test_open_ended_uptake_cannot_extend_outcome_coverage(postgres_event_connection, outcome_day):
    from tests.test_compliance_artifacts import check_uptake_outcome_coverage

    check_uptake_outcome_coverage(outcome_day, warehouse_connection=postgres_event_connection)


def test_retention_breakout_cohorts_match_all_ingress_paths(postgres_event_connection):
    from tests.parity_harness.cases import _retention_breakout_cohorts_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _retention_breakout_cohorts_case(
        warehouse_connection=postgres_event_connection,
        dialect="postgres",
    )
    assert_parity(case, run_case(case))


def test_automatic_exact_multi_arm_matches_all_ingress_paths(postgres_event_connection):
    """An automatically registered three-arm exact Bernoulli primary retains the
    same per-arm checkpoint and rows through both warehouse readers on live
    PostgreSQL as through the dataframe and moments constructors."""
    from tests.parity_harness.cases import _sequential_automatic_multi_arm_exact_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _sequential_automatic_multi_arm_exact_case(
        warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case))


@pytest.mark.parametrize("kind", ["always_valid", "asymptotic_mean"])
def test_automatic_segments_keep_the_warehouse_refusal_on_all_ingress_paths(
    postgres_event_connection, kind
):
    """A plan-predeclared segment family binds on every constructor; the frame
    constructors match while both warehouse readers on live PostgreSQL still
    refuse the segmented roster at capture with the relational segment-capture
    reason (no warehouse segment support is claimed)."""
    from tests.parity_harness.cases import (
        _sequential_automatic_breakout_continuous_case,
        _sequential_automatic_breakout_discrete_case,
    )
    from tests.parity_harness.runner import assert_parity, run_case

    builder = (
        _sequential_automatic_breakout_discrete_case
        if kind == "always_valid"
        else _sequential_automatic_breakout_continuous_case
    )
    case = builder(warehouse_connection=postgres_event_connection, dialect="postgres")
    assert_parity(case, run_case(case))


def test_scoped_source_snapshot_keeps_only_enrolled_in_window_rows(postgres_con):
    import ibis
    import pyarrow as pa

    from increment.query.session import SourceScope, WarehouseSession

    con = postgres_con
    con.create_table(
        "scoped_events",
        obj=pa.table(
            {
                "unit_id": ["u1", "u2", "u1", "outside"],
                "ts": [dt.datetime(2025, 1, 2)] * 2
                + [dt.datetime(2024, 12, 1), dt.datetime(2025, 1, 2)],
                "value": [10.0, 20.0, 1000.0, 2000.0],
            }
        ),
        temp=True,
    )
    enrolled = con.create_table(
        "scoped_enrolled", obj=pa.table({"unit_id": ["u1", "u2"]}), temp=True
    ).mutate(experiment_id=ibis.literal("exp"))
    session = WarehouseSession(con, Definitions(dialect="postgres"))
    sql = "SELECT * FROM scoped_events"
    pinned = session.pin_sources(
        (sql,),
        column_hints={sql: ("unit_id", "ts")},
        scope=SourceScope(enrolled_units=enrolled, window_start=dt.datetime(2025, 1, 1)),
    )
    try:
        with con.raw_sql("UPDATE scoped_events SET value = 9999"):
            pass
        actual = con.to_pyarrow(pinned.source_sql(sql)).to_pylist()
        assert sorted((row["unit_id"], row["value"]) for row in actual) == [
            ("u1", 10.0),
            ("u2", 20.0),
        ]
    finally:
        pinned.drop_materialized()


def test_bucketed_digest_matches_duckdb_and_detects_mutation(postgres_con):
    import math

    import ibis
    import pyarrow as pa

    from increment.query.artifact_digest import digest_relation_bucketed

    rows = pa.table(
        {
            "unit_id": ["u0", "u1", "u2", "u3", "u4"],
            "value": [1e20, math.nextafter(1e20, math.inf), -0.0, 2.0, None],
            "note": ["é", "", None, "plain", "nullable"],
            "ts": [dt.datetime(2025, 1, 1, 2, 3, 4, 567890, tzinfo=dt.UTC)] * 5,
        }
    )
    schema = (
        ("unit_id", "STRING", False),
        ("value", "FLOAT64", True),
        ("note", "STRING", True),
        ("ts", "TIMESTAMP_UTC_US", False),
    )
    oracle_con = ibis.duckdb.connect()
    try:
        oracle_table = oracle_con.create_table("digest_values", obj=rows)
        oracle = digest_relation_bucketed(
            oracle_con, oracle_table, "unit_covariate", schema, primary_key=("unit_id",)
        )
        table = postgres_con.create_table(
            "digest_values", obj=rows.take([4, 3, 2, 1, 0]), temp=True
        )
        actual = digest_relation_bucketed(
            postgres_con, table, "unit_covariate", schema, primary_key=("unit_id",)
        )
        assert actual.content_sha256 == oracle.content_sha256
        assert actual.schema_sha256 == oracle.schema_sha256
        assert actual.row_count == oracle.row_count == 5
        with postgres_con.raw_sql("UPDATE digest_values SET value = 3 WHERE unit_id = 'u3'"):
            pass
        changed = digest_relation_bucketed(
            postgres_con, table, "unit_covariate", schema, primary_key=("unit_id",)
        )
        assert changed.content_sha256 != actual.content_sha256
    finally:
        oracle_con.disconnect()


def test_artifact_abort_after_manifest_leaves_nothing_visible(postgres_con):
    run_artifact_abort_probe(postgres_con, dialect="postgres")


def test_artifact_explicit_catalog_path(postgres_con):
    """PostgreSQL has no scratch catalog, so this is the schema-only path."""
    run_artifact_explicit_catalog_probe(postgres_con, dialect="postgres")


def _postgres_mixed_cuped_rows(n_per_arm: int = 48):
    rows = []
    for arm_index, arm in enumerate(("control", "treatment", "withheld")):
        for i in range(n_per_arm):
            rows.append(
                {
                    "user_id": f"{arm}-{i}",
                    "variant": arm,
                    "date": dt.date(2025, 1, 10),
                    "revenue": float(4 + (i % 7) + arm_index * 1.25 + (i % 3) * 0.17),
                    "orders": 0.0 if arm == "withheld" else float(1 + i % 4),
                    "pre_revenue": None if i % 9 == 0 else float(8 + i % 13 + arm_index),
                    "country": ("north", "south", "west")[i % 3],
                    "device": ("mobile", "desktop")[i % 2],
                    "new_user": "new" if i % 4 == 0 else "existing",
                    "signups": float((i * 7 + arm_index) % 9),
                    "region": None if i % 6 == 0 else ("east", "west")[i % 2],
                }
            )
    return rows


@pytest.mark.timeout(1200)  # Native and adopted QR breakout parity needs extra time.
@pytest.mark.parametrize("route", ["categorical_cells", "general_qr"])
def test_postgres_mixed_cuped_definitions_and_artifact_match_local_and_duckdb(  # noqa: C901, PLR0915
    postgres_con, tmp_path, monkeypatch, route
):
    """Both warehouse fitters preserve mixed CUPED semantics without unit transfers."""
    import math

    import ibis
    import pyarrow as pa
    import yaml

    from increment import Analysis
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.plan import AnalysisPlan
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics.design import Randomized
    from increment.semantics.loader import load
    from tests.parity_harness.matrix_data import frame
    from tests.test_warehouse_cuped_qr import _guard_no_unit_keyed_transfers
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    event_relation = '"events mixed"'
    units = _postgres_mixed_cuped_rows(n_per_arm=96)
    if route == "general_qr":
        for row in units:
            if int(row["user_id"].rsplit("-", 1)[1]) % 5 == 0:
                row["pre_revenue"] = math.nan
    events = mixed_cuped_event_rows(units)
    if route == "general_qr":
        events = pa.Table.from_pylist(events)
        assert (
            sum(
                value is not None and math.isnan(value)
                for value in events.column("pre_revenue").to_pylist()
            )
            > 0
        )
    definitions_data = yaml.safe_load(
        mixed_cuped_definitions_yaml(event_relation, breakout=True).replace(
            "dialect: duckdb", "dialect: postgres"
        )
    )
    experiment_def = definitions_data["experiments"][0]
    experiment_def["allocation"] = {"control": 1 / 3, "treatment": 1 / 3, "withheld": 1 / 3}
    categorical_names = ("country", "device", "new_user")
    covariates = [
        {"name": name, "kind": "categorical", "property": name, "source": "events"}
        for name in categorical_names
    ]
    if route == "general_qr":
        covariates.insert(
            0,
            {
                "name": "pre_revenue",
                "kind": "continuous",
                "property": "pre_revenue",
                "source": "events",
                "missing": "impute-indicator",
            },
        )
    plan = experiment_def["plan"]
    for role in ("primary", "guardrails", "secondaries"):
        entries = plan[role]
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            entry["decision_method"]["covariates"] = covariates
    definitions_path = tmp_path / f"postgres-mixed-{route}.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions_data, sort_keys=False))

    categorical = [CupedCovariate(name=name, kind="categorical") for name in categorical_names]
    local_covariates = (
        categorical
        if route == "categorical_cells"
        else [
            CupedCovariate(name="pre_revenue", kind="continuous", missing="impute-indicator"),
            *categorical,
        ]
    )
    method = Method.cuped_multicovariate(covariates=local_covariates)
    local = Analysis.from_unit_summary(
        frame(units),
        unit="user_id",
        group="variant",
        metrics=[
            MetricSpec(name="revenue", decision_method=method),
            MetricSpec(
                name="rpo",
                type="ratio",
                numerator="revenue",
                denominator="orders",
                preferred_direction="increase",
                decision_method=method,
            ),
        ],
        plan=AnalysisPlan(primary="revenue", guardrails=("rpo",)),
        design=Randomized(
            control_group="control",
            allocation={"control": 1 / 3, "treatment": 1 / 3, "withheld": 1 / 3},
        ),
        breakouts=("region",),
    )
    duckdb = ibis.duckdb.connect()
    duckdb.create_table("events mixed", obj=events)
    postgres_con.create_table("events mixed", obj=events, overwrite=True)
    postgres_source = Analysis.from_definitions("mixed_cuped", definitions_path, postgres_con)
    duckdb_source = Analysis.from_definitions("mixed_cuped", definitions_path, duckdb)
    definitions = load(definitions_path)
    experiment = definitions.experiment("mixed_cuped")
    assert experiment is not None
    artifact_schema = f"cuped_{uuid4().hex[:12]}"
    store = WarehouseArtifactStore(postgres_con, schema_name=artifact_schema)
    context = artifact_context(definitions, experiment, "error")
    extensions = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    artifact_ref = postgres_source.publish_unit_day_artifact(store, extensions=extensions)
    adopted = Analysis.from_unit_day_artifact(store, artifact_ref, expected_context=context)

    _guard_no_unit_keyed_transfers(monkeypatch)
    from increment.errors import CodedError, IncrementWarning
    from increment.query.session import WarehouseSession

    transferred = []
    fetch_arrow = WarehouseSession.fetch_arrow

    def bounded_fetch(session, expression, **kwargs):
        result = fetch_arrow(session, expression, **kwargs)
        names = tuple(result.schema.names)
        assert "user_id" not in names and "unit_id" not in names
        assert result.num_rows <= 4096
        transferred.append((names, result.num_rows))
        return result

    monkeypatch.setattr(WarehouseSession, "fetch_arrow", bounded_fetch)

    def qr_scratch_names():
        with postgres_con.raw_sql(
            "SELECT tablename FROM pg_catalog.pg_tables "
            "WHERE schemaname = (SELECT nspname FROM pg_catalog.pg_namespace "
            "WHERE oid = pg_my_temp_schema()) "
            "AND left(tablename, 10) = '_cuped_qr_'"
        ) as cursor:
            return tuple(sorted(row[0] for row in cursor.fetchall()))

    def run_with_expected_withholding(analysis, *, breakout=False):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IncrementWarning)
            if breakout:
                rows = analysis.run_breakout(metrics=["revenue", "rpo"])
            else:
                rows = analysis.run(metrics=["revenue", "rpo"])
        if not breakout:
            assert any(warning.message.code == "readouts.run.cell_refused" for warning in caught)
        return rows

    scratch_before = qr_scratch_names()
    if route == "general_qr":
        failed = Analysis.from_definitions("mixed_cuped", definitions_path, postgres_con)
        create_table = postgres_con.create_table
        create_count = 0

        def fail_second_qr_relation(name, *args, **kwargs):
            nonlocal create_count
            if name.startswith("_cuped_qr_"):
                create_count += 1
                if create_count == 2:
                    raise RuntimeError("injected PostgreSQL QR scratch failure")
            return create_table(name, *args, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(postgres_con, "create_table", fail_second_qr_relation)
                with pytest.raises(CodedError) as refusal:
                    failed.run(metrics=["revenue"])
                assert refusal.value.code == "query.session.snapshot.materialization_failed"
            assert create_count == 2
            assert qr_scratch_names() == scratch_before
        finally:
            failed.close()

    results = []
    try:
        assert qr_scratch_names() == scratch_before
        for analysis in (local, duckdb_source, postgres_source, adopted):
            rows = run_with_expected_withholding(analysis)
            if analysis in (postgres_source, adopted):
                assert qr_scratch_names() == scratch_before
            results.append({(row.metric, row.group_id, row.method_role): row for row in rows})
        assert transferred
        assert all(result.keys() == results[0].keys() for result in results[1:]), [
            (sorted(set(result) - set(results[0])), sorted(set(results[0]) - set(result)))
            for result in results[1:]
        ]
        assert {key[0] for key in results[0]} == {"revenue", "rpo"}
        assert any(
            key[0] == "rpo" and key[1] == "withheld" and row.sampling_reason_code is not None
            for key, row in results[0].items()
        )
        for key, expected in results[0].items():
            for result in results[1:]:
                actual = result[key]
                assert actual.sampling_reason_code == expected.sampling_reason_code
                if expected.lift is None:
                    assert actual.lift is None
                    continue
                assert actual.lift is not None
                for field in ("value", "log_se", "lb", "ub"):
                    assert getattr(actual.lift, field) == pytest.approx(
                        getattr(expected.lift, field), rel=1e-9, abs=1e-11
                    )
                assert actual.dof == pytest.approx(expected.dof, rel=1e-9, abs=1e-11)

        # Breakout checks include a true SQL NULL segment and preserve the local fit.
        local_breakout = run_with_expected_withholding(local, breakout=True)
        assert any(
            row.metric == "rpo" and row.group_id == "withheld" and row.sampling_reason_code
            for row in local_breakout
        )
        for analysis in (duckdb_source, postgres_source, adopted):
            actual_rows = run_with_expected_withholding(analysis, breakout=True)
            assert any(
                row.metric == "rpo" and row.group_id == "withheld" and row.sampling_reason_code
                for row in actual_rows
            )
            expected = {
                (row.metric, row.dimension_value, row.group_id): row for row in local_breakout
            }
            actual = {(row.metric, row.dimension_value, row.group_id): row for row in actual_rows}
            assert actual.keys() == expected.keys()
            assert {row.dimension_value for row in actual_rows} >= {"__null__"}
            for key, expected_row in expected.items():
                actual_row = actual[key]
                assert actual_row.sampling_reason_code == expected_row.sampling_reason_code
                if expected_row.lift is not None:
                    assert actual_row.lift is not None
                    for field in ("value", "log_se", "lb", "ub"):
                        assert getattr(actual_row.lift, field) == pytest.approx(
                            getattr(expected_row.lift, field), rel=1e-9, abs=1e-11
                        )
                    assert actual_row.dof == pytest.approx(expected_row.dof, rel=1e-9, abs=1e-11)
    finally:
        local.close()
        duckdb_source.close()
        postgres_source.close()
        adopted.close()
        from integration.warehouse_execution._suite import _drop_scratch_namespace

        duckdb.disconnect()
        postgres_con.drop_table("events mixed", force=True)
        _drop_scratch_namespace(postgres_con, artifact_schema)


def test_postgres_cuped_without_temp_privilege_refuses_before_unit_transfer(
    postgres_con, tmp_path, monkeypatch
):
    """A real role without TEMP cannot bypass required QR scratch ownership."""
    import os

    import ibis
    import psycopg
    import yaml

    from increment import Analysis
    from increment.errors import CodedError
    from increment.query.session import WarehouseSession
    from tests.parity_harness.matrix_data import linear_cuped_rows
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    suffix = uuid4().hex[:10]
    database = f"cuped_no_temp_{suffix}"
    role = f"cuped_role_{suffix}"
    password = uuid4().hex
    env = os.environ
    admin = psycopg.connect(
        host=env["INCREMENT_TEST_POSTGRES_HOST"],
        port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
        user=env["INCREMENT_TEST_POSTGRES_USER"],
        password=env["INCREMENT_TEST_POSTGRES_PASSWORD"],
        dbname=env["INCREMENT_TEST_POSTGRES_DB"],
        autocommit=True,
    )
    database_admin = restricted = analysis = None
    try:
        admin.execute(f'CREATE DATABASE "{database}"')
        admin.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
        admin.execute(f'REVOKE TEMPORARY ON DATABASE "{database}" FROM PUBLIC')
        database_admin = ibis.postgres.connect(
            host=env["INCREMENT_TEST_POSTGRES_HOST"],
            port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
            user=env["INCREMENT_TEST_POSTGRES_USER"],
            password=env["INCREMENT_TEST_POSTGRES_PASSWORD"],
            database=database,
        )
        restricted = ibis.postgres.connect(
            host=env["INCREMENT_TEST_POSTGRES_HOST"],
            port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
            user=role,
            password=password,
            database=database,
        )
        assert database_admin is not None and restricted is not None
        units = linear_cuped_rows()
        database_admin.create_table("events", obj=mixed_cuped_event_rows(units))
        with database_admin.raw_sql(f'GRANT SELECT ON TABLE public.events TO "{role}"'):
            pass
        assert (
            admin.execute(
                "SELECT has_database_privilege(%s, %s, 'TEMP')",
                (role, database),
            ).fetchone()[0]
            is False
        )

        data = yaml.safe_load(
            mixed_cuped_definitions_yaml("events").replace("dialect: duckdb", "dialect: postgres")
        )
        definitions_path = tmp_path / "no-temp-cuped.yaml"
        definitions_path.write_text(yaml.safe_dump(data, sort_keys=False))
        analysis = Analysis.from_definitions("mixed_cuped", definitions_path, restricted)

        transfers = []
        fetch_arrow = WarehouseSession.fetch_arrow

        def count_transfer(session, expression, **kwargs):
            result = fetch_arrow(session, expression, **kwargs)
            transfers.append(tuple(result.schema.names))
            return result

        monkeypatch.setattr(WarehouseSession, "fetch_arrow", count_transfer)
        with pytest.raises(CodedError) as refusal:
            analysis.run(metrics=["revenue"])
        assert refusal.value.code == "query.session.snapshot.materialization_failed"
        assert not any("user_id" in names or "unit_id" in names for names in transfers)
        with database_admin.raw_sql(
            "SELECT count(*) FROM pg_catalog.pg_tables "
            "WHERE schemaname = (SELECT nspname FROM pg_catalog.pg_namespace "
            "WHERE oid = pg_my_temp_schema()) "
            "AND left(tablename, 10) = '_cuped_qr_'"
        ) as cursor:
            assert cursor.fetchone()[0] == 0
    finally:
        if analysis is not None:
            analysis.close()
        if restricted is not None:
            restricted.disconnect()
        if database_admin is not None:
            database_admin.disconnect()
        admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        admin.execute(f'DROP ROLE IF EXISTS "{role}"')
        admin.close()


def test_postgres_live_definitions_run_without_temp_privilege(postgres_con, tmp_path):  # noqa: PLR0915
    """Live first-invocation reads work without TEMP; captured reads still refuse."""
    import os
    from decimal import Decimal

    import ibis
    import psycopg
    import yaml

    from increment import Analysis
    from increment.errors import CodedError
    from increment.estimation.engine import Method
    from increment.query.session import WarehouseSession

    suffix = uuid4().hex[:10]
    database = f"live_no_temp_{suffix}"
    role = f"live_role_{suffix}"
    password = uuid4().hex
    env = os.environ
    admin = psycopg.connect(
        host=env["INCREMENT_TEST_POSTGRES_HOST"],
        port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
        user=env["INCREMENT_TEST_POSTGRES_USER"],
        password=env["INCREMENT_TEST_POSTGRES_PASSWORD"],
        dbname=env["INCREMENT_TEST_POSTGRES_DB"],
        autocommit=True,
    )
    database_admin = restricted = analysis = projected_live_analysis = projected_analysis = None
    try:
        admin.execute(f'CREATE DATABASE "{database}"')
        admin.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{password}'")
        admin.execute(f'REVOKE TEMPORARY ON DATABASE "{database}" FROM PUBLIC')
        database_admin = ibis.postgres.connect(
            host=env["INCREMENT_TEST_POSTGRES_HOST"],
            port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
            user=env["INCREMENT_TEST_POSTGRES_USER"],
            password=env["INCREMENT_TEST_POSTGRES_PASSWORD"],
            database=database,
        )
        restricted = ibis.postgres.connect(
            host=env["INCREMENT_TEST_POSTGRES_HOST"],
            port=int(env["INCREMENT_TEST_POSTGRES_PORT"]),
            user=role,
            password=password,
            database=database,
        )
        database_admin.create_table(
            "Évents",
            obj=[
                {
                    "user_id": Decimal(user_id),
                    "ts": dt.datetime(2025, 1, 10 if event == "purchase" else 1, 9),
                    "event": event,
                    "experiment_id": "exp",
                    "group_id": group,
                    "value": value,
                    "numeric_array": [Decimal("123456789012345678901234567890")],
                }
                for user_id, group, event, value in (
                    ("123456789012345678901234567890", "control", "enrolled", 0.0),
                    ("123456789012345678901234567890", "control", "purchase", 1.0),
                    ("223456789012345678901234567890", "treatment", "enrolled", 0.0),
                    ("223456789012345678901234567890", "treatment", "purchase", 2.0),
                )
            ],
            schema=ibis.schema(
                {
                    "user_id": "decimal(38, 0)",
                    "ts": "timestamp",
                    "event": "string",
                    "experiment_id": "string",
                    "group_id": "string",
                    "value": "float64",
                    "numeric_array": "array<decimal(38, 0)>",
                }
            ),
        )
        with database_admin.raw_sql('ALTER TABLE public."Évents" ALTER COLUMN event TYPE CHAR(16)'):
            pass
        with database_admin.raw_sql("CREATE SCHEMA analytics"):
            pass
        with database_admin.raw_sql(f'GRANT SELECT ON TABLE public."Évents" TO "{role}"'):
            pass
        with database_admin.raw_sql(f'GRANT USAGE ON SCHEMA analytics TO "{role}"'):
            pass
        with restricted.raw_sql("SET search_path TO analytics, public"):
            pass
        assert not admin.execute(
            "SELECT has_database_privilege(%s, %s, 'TEMP')", (role, database)
        ).fetchone()[0]

        definitions = {
            "dialect": "postgres",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM Évents /* plain source */",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "enrolled", "column": None},
                        {"name": "purchase", "column": "value"},
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "enrolled"}],
            "metrics": [
                {
                    "name": "purchase_rate",
                    "type": "conversion",
                    "entity": "user_id",
                    "fact": "purchase",
                    "window_days": 7,
                }
            ],
            "experiments": [
                {
                    "name": "exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "end": "2025-01-01",
                    "observation_end": "2025-01-10",
                    "control_group": "control",
                    "plan": {"secondaries": ["purchase_rate"]},
                }
            ],
        }
        definitions_path = tmp_path / "live-no-temp.yaml"
        definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))
        analysis = Analysis.from_definitions("exp", definitions_path, restricted)

        # This must be the first warehouse operation; SQL schema introspection
        # must not need to create a temporary view behind the caller's back.
        result = analysis.run(coherence="live", decision_method=Method(name="unadjusted"))
        assert result
        with pytest.raises(CodedError) as refusal:
            analysis.run(coherence="captured", decision_method=Method(name="unadjusted"))
        assert refusal.value.code == "query.session.snapshot.materialization_failed"
        projected_definitions = {
            **definitions,
            "fact_sources": [
                {
                    **definitions["fact_sources"][0],
                    "sql": (
                        "SELECT user_id, ts, event, experiment_id, group_id, value, numeric_array "
                        "FROM Évents /* projected source */"
                    ),
                }
            ],
        }
        projected_live_path = tmp_path / "live-no-temp-projected.yaml"
        projected_live_path.write_text(yaml.safe_dump(projected_definitions, sort_keys=False))
        projected_live_analysis = Analysis.from_definitions("exp", projected_live_path, restricted)
        assert projected_live_analysis.run(
            coherence="live", decision_method=Method(name="unadjusted")
        )
        array_session = WarehouseSession(restricted, defs=Definitions(dialect="postgres"))
        array_source = array_session.source_sql(projected_definitions["fact_sources"][0]["sql"])
        array_values = restricted.to_pyarrow(array_source.select("numeric_array"))
        assert array_values.column("numeric_array").to_pylist()[0] == [
            Decimal("123456789012345678901234567890")
        ]

        captured_definitions = {
            **projected_definitions,
            "experiments": [dict(definitions["experiments"][0])],
        }
        captured_definitions["experiments"][0].pop("end")
        captured_definitions["experiments"][0].pop("observation_end")
        projected_path = tmp_path / "captured-no-temp-projected.yaml"
        projected_path.write_text(yaml.safe_dump(captured_definitions, sort_keys=False))
        projected_analysis = Analysis.from_definitions("exp", projected_path, restricted)
        with pytest.raises(CodedError) as projected_refusal:
            projected_analysis.run(coherence="captured", decision_method=Method(name="unadjusted"))
        assert projected_refusal.value.code == "query.session.snapshot.materialization_failed"
    finally:
        if analysis is not None:
            analysis.close()
        if projected_analysis is not None:
            projected_analysis.close()
        if projected_live_analysis is not None:
            projected_live_analysis.close()
        if restricted is not None:
            restricted.disconnect()
        if database_admin is not None:
            database_admin.disconnect()
        admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        admin.execute(f'DROP ROLE IF EXISTS "{role}"')
        admin.close()


@pytest.mark.warehouse_postgres
def test_postgres_boolean_mean_metric_matches_duckdb_and_local_fit(postgres_con, tmp_path):
    """A boolean-valued mean traverses the native PostgreSQL readout and CUPED fit."""
    import ibis
    import yaml

    from increment import Analysis
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.plan import AnalysisPlan
    from increment.semantics.design import Randomized
    from tests.parity_harness.matrix_data import frame
    from tests.warehouse_cuped_cases import (
        mixed_cuped_definitions_yaml,
        mixed_cuped_event_rows,
    )

    units = [
        {**row, "active_value": (index % 3) != 0}
        for index, row in enumerate(_postgres_mixed_cuped_rows(96))
        if row["variant"] != "withheld"
    ]
    events = mixed_cuped_event_rows(units)
    for event in events:
        event["active_value"] = None
    for row in units:
        events.append(
            {
                **dict.fromkeys(events[0]),
                "user_id": row["user_id"],
                "ts": dt.datetime(2025, 1, 11, 9),
                "event": "active",
                "active_value": row["active_value"],
            }
        )

    definitions = yaml.safe_load(
        mixed_cuped_definitions_yaml('"events boolean"').replace(
            "dialect: duckdb", "dialect: postgres"
        )
    )
    definitions["fact_sources"][0]["facts"].append({"name": "active", "column": "active_value"})
    definitions["metrics"].append(
        {
            "name": "active_rate",
            "type": "mean",
            "entity": "user_id",
            "fact": "active",
            "aggregation": "sum",
            "window_days": 2,
        }
    )
    experiment = definitions["experiments"][0]
    experiment["plan"]["primary"]["metric"] = "active_rate"
    experiment["plan"]["primary"]["decision_method"]["covariates"] = [
        {"name": "country", "kind": "categorical", "property": "country", "source": "events"}
    ]
    definitions_path = tmp_path / "postgres-boolean-mean.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))

    method = Method.cuped_multicovariate(
        covariates=[CupedCovariate(name="country", kind="categorical")]
    )
    local = Analysis.from_unit_summary(
        frame(units),
        unit="user_id",
        group="variant",
        metrics=[MetricSpec(name="active_value", type="mean", decision_method=method)],
        plan=AnalysisPlan(primary="active_value"),
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
    )
    duckdb = ibis.duckdb.connect()
    duckdb.create_table("events boolean", obj=events)
    postgres_con.create_table("events boolean", obj=events, overwrite=True)
    native = Analysis.from_definitions("mixed_cuped", definitions_path, postgres_con)
    oracle = Analysis.from_definitions("mixed_cuped", definitions_path, duckdb)
    try:
        actual = {}
        for name, analysis in (("local", local), ("duckdb", oracle), ("postgres", native)):
            metric_name = "active_value" if name == "local" else "active_rate"
            rows = analysis.run(metrics=[metric_name])
            actual[name] = {row.group_id: row for row in rows}
        assert actual["postgres"].keys() == actual["duckdb"].keys() == actual["local"].keys()
        for group_id, expected in actual["local"].items():
            for name in ("duckdb", "postgres"):
                row = actual[name][group_id]
                assert row.lift is not None and expected.lift is not None
                assert row.lift.value == pytest.approx(expected.lift.value, rel=1e-9, abs=1e-11)
                assert row.dof == pytest.approx(expected.dof, rel=1e-9, abs=1e-11)
    finally:
        local.close()
        oracle.close()
        native.close()
        duckdb.disconnect()
        postgres_con.drop_table("events boolean", force=True)
