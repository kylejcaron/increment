"""The dashboard accepts interaction evidence only for its pinned focal generation."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import file_digest

import ibis
import pyarrow as pa
import pytest

from increment import from_definitions, load_definitions
from increment.dashboard import DashboardConfig, dashboard_html, prepare_dashboard, readout_csv
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
from tests.test_dashboard_capture import (
    EXPERIMENTS,
    FACT_SOURCES,
    _event_rows,
    _page_payload,
    _write_definitions,
)

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
        observed_inventory=ObservedInventory.model_validate({**roster, "cutoff": _CUTOFF}),
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


def test_binding_rejects_even_a_full_table_sample(tmp_path):
    definitions = _write_definitions(
        tmp_path / "definitions",
        facts=FACT_SOURCES.replace(
            "FROM analytics.event_log", "FROM analytics.event_log TABLESAMPLE SYSTEM (100 PERCENT)"
        ),
    )
    path = tmp_path / "facts.duckdb"
    _publish(path)
    capture = _empty_capture(definitions)
    with _reader(path, definitions) as (experiment, reader, con):
        manifest = _manifest(path, con, experiment, capture)
        with pytest.raises(CodedError) as error:
            bind_interaction_scan(
                capture, source=InteractionDuckDBSource(experiment, reader, manifest)
            )
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "focal_facts_not_verified"


def test_manifest_fingerprint_survives_reader_timezone_change(tmp_path):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    _publish(path)
    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql("ALTER TABLE analytics.event_log ADD COLUMN archived_at TIMESTAMPTZ")
        con.raw_sql("UPDATE analytics.event_log SET archived_at = '2025-05-02 00:00:00+00'")
        con.raw_sql("CHECKPOINT")
    finally:
        con.disconnect()
    capture = _empty_capture(definitions)
    with _reader(path, definitions) as (experiment, reader, con):
        con.con.execute("SET TimeZone = 'UTC'")
        manifest = _manifest(path, con, experiment, capture)
        con.con.execute("SET TimeZone = 'America/Los_Angeles'")
        bound = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        dashboard = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=bound)
        assert dashboard.interactions is not None
        assert con.con.execute("SELECT current_setting('TimeZone')").fetchone() == (
            "America/Los_Angeles",
        )


def _joint_capture(definitions):
    from increment.overlap import CombinationRegistration, InteractionCombination

    original = _empty_capture(definitions)
    scope = original.plan.scope
    focal = scope.focal.entry
    others = tuple(
        focal.model_copy(
            update={
                "identity": ExperimentIdentity(target=name, experiment_id=name, revision="v1"),
                "display_name": name,
            }
        )
        for name in ("Cards", "Filters")
    )
    combination = InteractionCombination(
        candidates=tuple(entry.identity for entry in others),
        joint_assignment="independent_unit_bernoulli",
    )
    registration = CombinationRegistration(
        registered_at=_ENTRY - timedelta(days=1),
        scope=scope,
        metrics=original.plan.metrics,
        combinations=(combination,),
    )
    inventory = original.plan.inventory.model_copy(update={"entries": (focal, *others)})
    plan = original.plan.model_copy(
        update={
            "inventory": inventory,
            "combination_registration": registration,
        }
    )
    assert original.observed_inventory is not None
    observed = original.observed_inventory.model_copy(
        update={
            "entries": inventory.entries,
            "combination_registration_sha256": registration.registration_id,
        }
    )
    return capture_interaction_scan(plan, observed_inventory=observed, evidence_by_study={})


def _joint_manifest(path, con, experiment, capture):
    from increment.overlap import CombinationFeed

    manifest = _manifest(path, con, experiment, capture)
    catalog = manifest.focal_assignments.coordinate[0]
    feed = CombinationFeed(
        candidates=capture.plan.combinations[0].candidates,
        roster=interaction_duckdb_fact(con.table("joint_roster", database=(catalog, "analytics"))),
        outcomes=interaction_duckdb_fact(
            con.table("joint_outcomes", database=(catalog, "analytics"))
        ),
    )
    return manifest.model_copy(update={"combination_feeds": (feed,)})


def test_joint_roster_keeps_pending_missing_and_zero_outcomes_distinct(tmp_path):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    _publish(path)
    capture = _joint_capture(definitions)
    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql("""
            CREATE TABLE analytics.joint_roster AS
            SELECT * FROM (VALUES ('off','000'),('on','100'),('late','001'),('lost','101'))
                AS t(unit_id, assignment)
            """)
        con.raw_sql("""
            CREATE TABLE analytics.joint_outcomes(
                unit_id VARCHAR, metric VARCHAR, definition_sha256 VARCHAR,
                value DOUBLE, finalized_at TIMESTAMPTZ)
            """)
        metric = capture.plan.metrics[0]
        con.con.executemany(
            "INSERT INTO analytics.joint_outcomes VALUES (?, ?, ?, ?, ?)",
            [
                (unit, metric.name, metric.definition_sha256, value, when)
                for unit, value, when in (
                    ("off", 0.0, _CUTOFF),
                    ("on", 1.0, _CUTOFF),
                    ("late", None, None),
                )
            ],
        )
        con.raw_sql("CHECKPOINT")
    finally:
        con.disconnect()

    with _reader(path, definitions) as (experiment, reader, con):
        manifest = _joint_manifest(path, con, experiment, capture)
        bound = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        snapshot = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=bound)
        assert snapshot.interactions is not None
        (result,) = snapshot.interactions.combinations
        cells = {cell.cell: cell for cell in result.cells}
        assert (cells["000"].n_observed, cells["000"].n_success) == (1, 0)
        assert (cells["001"].n_pending, cells["101"].n_missing) == (1, 1)
        assert result.rows[0].estimate == 1.0
        assert result.rows[1].estimate is None
        assert {(reason.code, reason.cell) for reason in result.rows[1].reasons} == {
            ("pending_outcomes", "001"),
            ("missing_outcomes", "101"),
        }
        payload = _page_payload(dashboard_html(experiment, snapshot=snapshot))
        packet = payload["interactions"]["metrics"][metric.name]
        combination_view = packet["combinations"][0]
        assert 'aria-label="Cards off"' in combination_view["effects"]
        assert 'aria-label="Filters on"' in combination_view["effects"]
        assert "inc-assignment-dot" not in combination_view["change"]
        assert combination_view["effects"].count("<svg ") == 2  # one point and its shared axis
        (feed,) = manifest.combination_feeds
        wrong_order = feed.model_copy(update={"candidates": feed.candidates[::-1]})
        with pytest.raises(CodedError) as mismatch:
            bind_interaction_scan(
                capture,
                source=InteractionDuckDBSource(
                    experiment,
                    reader,
                    manifest.model_copy(update={"combination_feeds": (wrong_order,)}),
                ),
            )
        assert mismatch.value.code == "overlap.scan.invalid"
        assert mismatch.value.context["reason"] == "unknown_combination"


@pytest.mark.parametrize("receipt", ["matching", "wrong_hash", "wrong_timestamp"])
def test_only_matching_native_registration_enables_adjusted_joint_intervals(tmp_path, receipt):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    _publish(path)
    capture = _joint_capture(definitions)
    registration = capture.plan.combination_registration
    assert registration is not None
    metric = capture.plan.metrics[0]
    con = ibis.duckdb.connect(path)
    try:
        con.con.execute("""
            CREATE TABLE analytics.joint_roster AS
            SELECT concat(assignment, '-', sample) AS unit_id, assignment
            FROM unnest(['000', '001', '010', '011', '100', '101', '110', '111'])
                AS cells(assignment)
            CROSS JOIN range(80) AS units(sample)
            """)
        con.con.execute(
            """
            CREATE TABLE analytics.joint_outcomes AS
            SELECT unit_id, ? AS metric, ? AS definition_sha256, ?::TIMESTAMPTZ AS finalized_at,
                   (CAST(split_part(unit_id, '-', 2) AS BIGINT) <
                        CASE WHEN starts_with(assignment, '1') THEN 60 ELSE 20 END)::DOUBLE AS value
            FROM analytics.joint_roster
            """,
            [metric.name, metric.definition_sha256, _CUTOFF],
        )
        con.con.execute("""
            CREATE TABLE main.increment_interaction_combination_registration (
                registration_sha256 VARCHAR, registered_at TIMESTAMPTZ
            )
            """)
        con.con.execute(
            "INSERT INTO main.increment_interaction_combination_registration VALUES (?, ?)",
            [
                "0" * 64 if receipt == "wrong_hash" else registration.registration_id,
                _ENTRY if receipt == "wrong_timestamp" else registration.registered_at,
            ],
        )
        con.raw_sql("CHECKPOINT")
    finally:
        con.disconnect()

    with _reader(path, definitions) as (experiment, reader, con):
        manifest = _joint_manifest(path, con, experiment, capture)
        bound = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, manifest)
        )
        snapshot = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=bound)
    assert snapshot.interactions is not None
    (result,) = snapshot.interactions.combinations
    first = result.rows[0]
    assert first.estimate is not None and first.estimate == pytest.approx(0.5)
    if receipt == "matching":
        assert first.lower is not None and first.upper is not None
        assert 0 < first.lower < first.estimate < first.upper
        assert first.decision == "detected"
    else:
        assert first.lower is first.upper is first.decision is None
        assert [reason.code for reason in first.reasons] == ["joint_registration_unverified"]
