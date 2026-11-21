"""The dashboard accepts interaction evidence only for its pinned focal generation."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

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
    bind_interaction_scan,
    capture_interaction_scan,
    freeze_interaction_scan,
    interaction_metric_sha256,
)
from increment.query.session import SourceSnapshotEvidence
from tests.test_dashboard_capture import _event_rows, _write_definitions

pytestmark = pytest.mark.integration

_START = datetime(2025, 1, 15, tzinfo=UTC)
_ENTRY = _START - timedelta(days=1)
_END = datetime(2025, 2, 15, tzinfo=UTC)
_CUTOFF = datetime(2025, 5, 2, tzinfo=UTC)
_CONFIG = DashboardConfig(expected_allocation={"control": 0.5, "treatment": 0.5})


@contextmanager
def _reader(path, definitions, *, readonly=True):
    con = ibis.duckdb.connect(path, read_only=readonly)
    try:
        experiment, reader = from_definitions(
            definitions,
            con,
            experiment_name="storefront_refresh",
            source_snapshot_evidence=SourceSnapshotEvidence(_CUTOFF, {"event_log": _CUTOFF}),
        )
        with reader:
            yield experiment, reader
    finally:
        con.disconnect()


def _empty_capture(definitions):
    metric = next(
        item for item in load_definitions(definitions).metrics if item.name == "checkout_conversion"
    )
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
        metrics=(
            OverlapMetric(
                name=metric.name,
                unit="rate",
                definition_sha256=interaction_metric_sha256(metric),
                kind="binary_mean",
                role="primary",
                better="higher",
            ),
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


def test_snapshot_rejects_same_count_data_from_another_sealed_generation(tmp_path):
    definitions = _write_definitions(tmp_path / "definitions")
    path = tmp_path / "facts.duckdb"
    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql("CREATE SCHEMA analytics")
        con.create_table("event_log", pa.Table.from_pylist(_event_rows(100)), database="analytics")
    finally:
        con.disconnect()
    capture = _empty_capture(definitions)
    with _reader(path, definitions) as (experiment, reader):
        attachment = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, {})
        )
        snapshot = prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
        headline = readout_csv(prepare_dashboard(experiment, reader, config=_CONFIG))
    assert snapshot.interactions is not None and snapshot.interactions.no_candidates
    assert readout_csv(snapshot) == headline

    with _reader(path, definitions) as (experiment, reader):
        restored = bind_interaction_scan(
            capture, source=InteractionDuckDBSource(experiment, reader, {})
        )
    assert restored.value.focal_receipt == attachment.value.focal_receipt

    con = ibis.duckdb.connect(path)
    try:
        con.raw_sql(
            "UPDATE analytics.event_log SET revenue = revenue + 1 WHERE revenue IS NOT NULL"
        )
    finally:
        con.disconnect()
    with _reader(path, definitions) as (experiment, reader):
        with pytest.raises(CodedError) as error:
            prepare_dashboard(experiment, reader, config=_CONFIG, interactions=attachment)
    assert error.value.code == "overlap.scan.invalid"
    assert error.value.context["reason"] == "different_source_generation"
    assert readout_csv(snapshot) == headline
