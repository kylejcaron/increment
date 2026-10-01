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

from increment import Analysis, Report
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
        postgres_con.drop_database(schema, force=True)


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
def test_artifact_publish_adopt_parity(postgres_con, monkeypatch, clustered, fail_adoption):
    con = postgres_con
    WarehouseArtifactStore(con)
    metadata = ("ud_manifest_index", "ud_manifest_dropped")
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
            Report.from_definitions(definitions, con)
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


def test_denied_temp_creation_fails_the_gate(postgres_con, monkeypatch):
    """A denied CREATE TEMP TABLE privilege must not be swallowed as a
    passing store='always' run. ``WarehouseSession.materialize_table``
    degrades a create failure to an unmaterialized live expression and
    warns; this suite's project-wide ``filterwarnings = ["error", ...]``
    promotes that warning to a hard failure, and the probe's own
    required-materialization check would catch it even if it didn't."""
    con = postgres_con
    create_table = con.create_table

    def deny_temp(name, *args, **kwargs):
        if kwargs.get("temp") is True:
            raise RuntimeError("simulated: permission denied for CREATE TEMP TABLE")
        return create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", deny_temp)
    with pytest.raises(
        (AssertionError, UserWarning), match="did not materialize|could not materialize"
    ):
        run_materialized_cache_correction_probe(con, dialect="postgres", monkeypatch=monkeypatch)


def test_partial_materialization_fails_the_gate(postgres_con, monkeypatch):
    """``store='always'`` must materialize BOTH the spine and
    revenue-stats TEMP tables on every readout. If only the spine lands
    (e.g. the backend denies just the stats create mid-flight), the
    gate must catch the partial materialization rather than treat spine
    alone as sufficient."""
    con = postgres_con
    create_table = con.create_table

    def deny_stats_temp(name, *args, **kwargs):
        if kwargs.get("temp") is True and name.startswith("exp_stats_revenue_"):
            raise RuntimeError("simulated: revenue-stats materialization denied")
        return create_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", deny_stats_temp)
    with pytest.raises((AssertionError, UserWarning), match="revenue stats|could not materialize"):
        run_materialized_cache_correction_probe(con, dialect="postgres", monkeypatch=monkeypatch)


def test_duplicate_spine_creation_without_stats_fails_the_gate(postgres_con, monkeypatch):
    """Two physical spine materializations cannot substitute for a
    missing revenue-stats materialization -- the two required roles are
    checked independently by name prefix, never satisfied by volume."""
    con = postgres_con
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


def test_extra_unrelated_temp_creation_does_not_fail_the_gate(postgres_con, monkeypatch):
    """An extra, unrelated TEMP table created alongside the required
    spine/revenue-stats materializations must not trip the required
    -role check: only the two named roles are required, extra
    relations are tolerated."""
    con = postgres_con
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
    from tests.parity_harness.cases import _lift_prior_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _lift_prior_case(
        kind, warehouse_connection=postgres_event_connection, dialect="postgres"
    )
    assert_parity(case, run_case(case))


def test_clearing_prior_matches_all_ingress_paths(postgres_event_connection):
    from tests.parity_harness.cases import _prior_reset_case
    from tests.parity_harness.runner import assert_parity, run_case

    case = _prior_reset_case(warehouse_connection=postgres_event_connection, dialect="postgres")
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
    assert_parity(case, run_case(case))


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
