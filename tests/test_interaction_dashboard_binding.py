"""The dashboard accepts interaction evidence only for its pinned focal generation."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import file_digest

import ibis
import pyarrow as pa
import pytest

from increment import from_definitions, load_definitions
from increment.dashboard import DashboardConfig, prepare_dashboard, readout_csv
from increment.errors import CodedError
from increment.overlap import (
    ExperimentIdentity,
    FocalStudy,
    InteractionDuckDBSource,
    InteractionInventory,
    InventoryEntry,
    ObservedInventory,
    OverlapMetric,
    SourceGenerationManifest,
    bind_interaction_scan,
    capture_interaction_scan,
    freeze_interaction_scan,
    interaction_duckdb_fact,
    interaction_metric_sha256,
)
from increment.query.session import SourceSnapshotEvidence
from tests.test_dashboard_capture import EXPERIMENTS, FACT_SOURCES, _event_rows, _write_definitions

pytestmark = pytest.mark.integration

_START = datetime(2025, 1, 15, tzinfo=UTC)
_ENTRY = _START - timedelta(days=1)
_END = datetime(2025, 2, 15, tzinfo=UTC)
_CUTOFF = datetime(2025, 5, 2, tzinfo=UTC)
_CONFIG = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})
_GENERATION = "storefront-2025-05-02"
_ASSIGNMENT_LEDGER = """
    assignment_ledger:
      sql: SELECT * FROM analytics.assignments
      roles:
        unit: subject
        experiment: experiment_key
        arm: arm_name
        version: revision
        period: period
        allocated_at: assigned_at
        exposed_at: exposure_at
        triggered_at: trigger_at
"""

_NATIVE_EXPOSURE = """
exposures:
  - name: first_page_view
    sql: SELECT * FROM analytics.enrollment
"""


@contextmanager
def _reader(path, definitions, *, watermark=_CUTOFF):
    con = ibis.duckdb.connect(path, read_only=True)
    try:
        experiment, reader = from_definitions(
            definitions,
            con,
            experiment_name="storefront_refresh",
            source_snapshot_evidence=SourceSnapshotEvidence(
                _CUTOFF, {"event_log": _CUTOFF, "purchases": watermark}
            ),
        )
        with reader:
            yield experiment, reader, con
    finally:
        con.disconnect()


def _empty_capture(definitions, names=("checkout_conversion",)):
    metrics = {item.name: item for item in load_definitions(definitions).metrics}
    entry = InventoryEntry(
        identity=ExperimentIdentity(
            target="storefront", experiment_id="storefront_refresh", revision="v1"
        ),
        archive_sha256="a" * 64,
        unit_namespace="user_id",
        active_from=_ENTRY,
        active_until=_END,
        arms=("control", "treatment"),
        display_name="Storefront refresh",
    )
    roster = {
        "universe_id": "storefront",
        "source_id": "synthetic-catalog",
        "version": "v1",
        "digest": "b" * 64,
        "covered_from": _ENTRY,
        "complete_through": _END,
        "coverage": "complete",
        "entries": (entry,),
    }
    plan = freeze_interaction_scan(
        inventory=InteractionInventory(frozen_at=_ENTRY - timedelta(days=1), **roster),
        focal=FocalStudy(
            entry=entry,
            target="jointly_eligible",
            population_id="eligible",
            eligibility_rule_sha256="c" * 64,
            enrollment_rule_sha256="d" * 64,
        ),
        metrics=tuple(
            OverlapMetric(
                name=name,
                unit="rate",
                definition_sha256=interaction_metric_sha256(metrics[name]),
                kind="binary_mean" if name == "checkout_conversion" else "mean",
                role="primary" if name == "checkout_conversion" else "secondary",
                better="higher",
            )
            for name in names
        ),
        entry_open=_ENTRY,
        start=_START,
        end=_END,
        cutoff=_CUTOFF,
        alpha=0.05,
    )
    return capture_interaction_scan(
        plan,
        observed_inventory=ObservedInventory(cutoff=_CUTOFF, **roster),
        evidence_by_study={},
    )


def _publish(path, *, generation=_GENERATION, split=False):
    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql("CREATE SCHEMA analytics")
        con.create_table("event_log", pa.Table.from_pylist(_event_rows(100)), database="analytics")
        if split:
            con.raw_sql("CREATE TABLE analytics.purchases AS SELECT * FROM analytics.event_log")
            con.raw_sql(
                "UPDATE analytics.purchases SET event_at = '2024-12-20' WHERE revenue IS NULL"
            )
        con.raw_sql(
            "CREATE TABLE main.increment_interaction_generation (version INTEGER, generation_id VARCHAR)"
        )
        con.con.execute(
            "INSERT INTO main.increment_interaction_generation VALUES (1, ?)", [generation]
        )
        con.raw_sql("CHECKPOINT")
    finally:
        con.disconnect()


def _manifest(path, con, experiment, capture, *, generation=_GENERATION, split=False):
    [(catalog,)] = con.con.execute("SELECT current_database()").fetchall()
    event_log = interaction_duckdb_fact(con.table("event_log", database=(catalog, "analytics")))
    outcomes = (
        interaction_duckdb_fact(con.table("purchases", database=(catalog, "analytics")))
        if split
        else event_log
    )
    with path.open("rb") as stream:
        dataset = file_digest(stream, "sha256").hexdigest()
    return SourceGenerationManifest(
        source_id=capture.plan.scope.source_id,
        generation_id=generation,
        dataset_sha256=dataset,
        cutoff=capture.plan.scope.cutoff,
        scan_plan_id=capture.plan.plan_id,
        definition_sha256=experiment.hash(),
        focal_assignments=event_log,
        focal_outcomes=tuple(
            (metric.name, outcomes if metric.name == "checkout_conversion" else event_log)
            for metric in capture.plan.metrics
        ),
        pair_feeds=(),
    )


def test_snapshot_rejects_same_count_data_from_another_sealed_generation(tmp_path):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    _publish(path)
    capture = _empty_capture(definitions)
    with _reader(path, definitions) as (experiment, reader, con):
        manifest = _manifest(path, con, experiment, capture)
        attachment = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        snapshot = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
        headline = readout_csv(prepare_dashboard(experiment, reader, config=_CONFIG))
        wrong_exposure = manifest.focal_assignments.model_copy(update={"content_sha256": "a" * 64})
        mismatched = manifest.model_copy(update={"focal_exposures": wrong_exposure})
        with pytest.raises(CodedError) as unknown_exposure:
            bind_interaction_scan(
                capture, source=InteractionDuckDBSource(experiment, reader, mismatched)
            )
        assert unknown_exposure.value.context["reason"] == "focal_exposures_not_verified"
    assert snapshot.interactions is not None and snapshot.interactions.no_candidates
    assert readout_csv(snapshot) == headline

    with _reader(path, definitions) as (experiment, reader, _):
        restored = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
    assert restored.value.focal_receipt == attachment.value.focal_receipt

    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql(
            "UPDATE analytics.event_log SET revenue = revenue + 1 WHERE revenue IS NOT NULL"
        )
    finally:
        con.disconnect()
    with _reader(path, definitions) as (experiment, reader, _):
        with pytest.raises(CodedError) as error:
            prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "different_source_generation"
    assert readout_csv(snapshot) == headline


def test_split_outcomes_keep_historical_rows_and_refuse_a_shorter_headline(tmp_path):
    facts = (
        FACT_SOURCES.replace(
            "      - name: purchase\n        column: revenue\n        description: A purchase with revenue\n",
            "",
        )
        + """
  - name: purchases
    sql: SELECT * FROM analytics.purchases
    timestamp_column: event_at
    entities: [user_id]
    facts:
      - {name: purchase, column: revenue}
    properties:
      - {name: country, column: country_code, dtype: string, as_of: static}
"""
    )
    definitions = _write_definitions(tmp_path / "definitions", facts=facts)
    path = tmp_path / "facts.duckdb"
    _publish(path, split=True)
    capture = _empty_capture(definitions, ("checkout_conversion", "session_seconds"))
    with _reader(path, definitions) as (experiment, reader, con):
        manifest = _manifest(path, con, experiment, capture, split=True)
        attachment = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        dashboard = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
        assert dashboard.interactions is not None
    with _reader(path, definitions, watermark=_END - timedelta(days=5)) as (experiment, reader, _):
        with pytest.raises(CodedError) as error:
            prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "focal_outcome_window_incomplete"


def test_manifest_cannot_identify_a_replaced_file_on_an_old_reader(tmp_path):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    newer = tmp_path / "new"
    newer.mkdir()
    replacement = newer / path.name
    _publish(path)
    _publish(replacement, generation="new-publication")
    capture = _empty_capture(definitions)
    with _reader(replacement, definitions) as (experiment, _, con):
        manifest = _manifest(replacement, con, experiment, capture, generation="new-publication")
    with _reader(path, definitions) as (experiment, reader, _):
        replacement.replace(path)
        with pytest.raises(CodedError) as error:
            bind_interaction_scan(
                capture, source=InteractionDuckDBSource(experiment, reader, manifest)
            )
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "different_source_generation"


def test_native_assignment_ledger_does_not_substitute_for_headline_exposures(tmp_path):
    specs = EXPERIMENTS.replace(
        "    allocation_scheme: independent",
        "    allocation_scheme: independent\n" + _ASSIGNMENT_LEDGER,
        1,
    )
    path = tmp_path / "facts.duckdb"
    _publish(path)
    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql(
            """CREATE TABLE analytics.enrollment AS
            SELECT user_id AS unit_id, event_at AS ts, group_id, experiment_id
            FROM analytics.event_log WHERE event = 'page_view' AND group_id IS NOT NULL"""
        )
        con.raw_sql(
            """CREATE TABLE analytics.assignments AS
            SELECT unit_id AS subject, experiment_id AS experiment_key, group_id AS arm_name,
            3 AS revision, 'global' AS period, ts AS assigned_at, ts AS exposure_at,
            NULL::TIMESTAMP AS trigger_at FROM analytics.enrollment"""
        )
        con.raw_sql("CREATE VIEW analytics.live_enrollment AS SELECT * FROM analytics.enrollment")
        con.raw_sql("CHECKPOINT")
    finally:
        con.disconnect()

    native = _write_definitions(tmp_path / "native", experiments=specs)
    (native / "exposures.yaml").write_text(_NATIVE_EXPOSURE)
    capture = _empty_capture(native)
    with _reader(path, native) as (experiment, reader, con):
        manifest = _manifest(path, con, experiment, capture)
        [(catalog,)] = con.con.execute("SELECT current_database()").fetchall()
        manifest = manifest.model_copy(
            update={
                "focal_assignments": interaction_duckdb_fact(
                    con.table("assignments", database=(catalog, "analytics"))
                ),
                "focal_exposures": interaction_duckdb_fact(
                    con.table("enrollment", database=(catalog, "analytics"))
                ),
            }
        )
        attachment = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        dashboard = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
        assert dashboard.interactions is not None

    live = _write_definitions(tmp_path / "live", experiments=specs)
    (live / "exposures.yaml").write_text(_NATIVE_EXPOSURE.replace("enrollment", "live_enrollment"))
    with _reader(path, live) as (experiment, reader, _):
        exact = manifest.model_copy(update={"definition_sha256": experiment.hash()})
        with pytest.raises(CodedError) as error:
            bind_interaction_scan(
                capture, source=InteractionDuckDBSource(experiment, reader, exact)
            )
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "focal_facts_not_verified"
