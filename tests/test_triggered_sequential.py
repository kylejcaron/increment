"""Registered sequential monitoring of the triggered population.

A trigger-declared capture retains two chains, assigned and triggered, each a
prefix of its own fixed reveal order. These tests pin the consumer-visible
contract: both populations report real checkpoint rows, every contributing
feed must be certified complete through the capture horizon, a process that
began without a triggered commitment never acquires one, assignment-anchored
uptake stays outside the triggered construction, and each population's family
is selected and labelled on its own chain with the same guarantee.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import pickle
from contextlib import AbstractContextManager
from dataclasses import replace
from fractions import Fraction
from typing import Any, cast

import ibis
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from increment import run, workflows
from increment.artifacts import SourceSnapshotEvidence, publish_unit_day_artifact
from increment.breakout.estimates import LiftEstimates
from increment.builders import from_definitions, from_moments, from_unit_day_artifact
from increment.errors import CapabilityError, CodedError
from increment.estimation.readout_types import ReadoutResults
from increment.evidence import (
    SequentialCaptureRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from increment.kernel.contracts import Scope
from increment.query.artifact_contract import unit_day_artifact_extension_catalog
from increment.query.artifact_publish import artifact_context
from increment.query.session import WarehouseArtifactStore
from increment.semantics import load
from increment.semantics.models import InferenceSpec
from increment.sequential_state import SequentialSnapshot
from increment.workflows import (
    capture_sequential,
    experiment_definition,
    export_moments,
    sequential_snapshot,
)
from tests.analysis_factory import bound_source, make_binding, view_readout
from tests.capability_consumers_auxiliary import assert_expected, assert_refusal

_UTC_US = pa.timestamp("us", tz="UTC")
_WINDOW_DAYS = 3
_START = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _definitions(
    *,
    plan: str,
    trigger: str = "saw_surface",
    trigger_sql: bool = False,
    encouragement: bool = False,
    secondaries: tuple[str, ...] = (),
    breakouts: bool = False,
    uptake_source: bool = False,
) -> str:
    trigger_exposure = (
        "  - name: saw_surface\n"
        "    sql: SELECT user_id AS unit_id, ts, group_id FROM events WHERE event = 'saw_surface'\n"
        if trigger_sql
        else "  - name: saw_surface\n    fact: saw_surface\n"
    )
    extra_metrics = "".join(
        f"  - name: {name}\n    type: mean\n    entity: user_id\n    fact: {name}\n"
        f"    aggregation: sum\n    window_days: {_WINDOW_DAYS}\n"
        for name in secondaries
    )
    extra_facts = "".join(f"      - name: {name}\n        column: value\n" for name in secondaries)
    breakout = (
        "    breakouts:\n      - source: events\n        property: region\n" if breakouts else ""
    )
    design = (
        "    design:\n"
        "      mechanism: encouragement\n"
        "      uptake:\n"
        "        fact: clicked\n"
        "        window_days: 2\n"
        "      one_sided: false\n"
        "      exclusion_restriction:\n"
        "        justification: the surface gates revenue\n"
        if encouragement
        else ""
    )
    uptake_fact = "" if uptake_source else "      - name: clicked\n        column: null\n"
    uptake_source_block = (
        "  - name: clicks\n"
        "    sql: SELECT * FROM events WHERE event = 'clicked'\n"
        "    timestamp_column: ts\n"
        "    entities: [user_id]\n"
        "    facts:\n"
        "      - name: clicked\n"
        "        column: null\n"
        if uptake_source
        else ""
    )
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    properties:
      - {{name: region, column: region, dtype: string, as_of: pre_exposure}}
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
{uptake_fact}{extra_facts}{uptake_source_block}exposures:
  - name: assignment
    fact: enrolled
{trigger_exposure}metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: {_WINDOW_DAYS}
{extra_metrics}experiments:
  - name: exp
    exposure: assignment
    trigger: {trigger}
    unit: user_id
    allocation: {{C: 0.5, T: 0.5}}
    allocation_scheme: independent
    start: 2024-01-01T00:00:00
    control_group: C
{design}{breakout}    plan:
{plan}
"""


_PRIMARY_PLAN = "      primary: revenue\n      alternative: two-sided\n      inference:\n        kind: asymptotic_mean\n"
_FAMILY_PLAN = (
    "      primary: revenue\n      secondaries: [aux_a, aux_b]\n      alternative: two-sided\n"
    "      inference:\n        kind: asymptotic_mean\n"
)


def _events(
    *,
    n_per_arm: int = 400,
    seed: int = 11,
    trigger_rate: float = 0.5,
    effect: float = 0.4,
    secondaries: tuple[str, ...] = (),
    secondary_effect: float = 0.0,
    uptake: bool = False,
    treatment_trigger_delay: int = 0,
) -> pa.Table:
    """Exposure days 1-6, trigger delay 0-3 days, outcome one day after the anchor.

    Only triggered treated units carry the effect, so the triggered contrast is
    the undiluted one and the assigned contrast is diluted by the trigger rate.
    """
    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "user_id": [],
        "group_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }

    def add(uid, arm, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        for i in range(n_per_arm):
            uid = f"{arm}{i:04d}"
            exposure = _START + dt.timedelta(days=int(rng.integers(1, 7)), hours=9)
            add(uid, arm, "enrolled", 0.0, exposure)
            triggered = rng.random() < trigger_rate
            anchor = exposure
            if triggered:
                delay = int(rng.integers(0, 4)) + (treatment_trigger_delay if arm == "T" else 0)
                anchor = exposure + dt.timedelta(days=delay, hours=1)
                add(uid, arm, "saw_surface", 0.0, anchor)
            if uptake and rng.random() < (0.7 if arm == "T" else 0.2):
                add(uid, arm, "clicked", 0.0, exposure + dt.timedelta(hours=2))
            base = float(rng.lognormal(0.0, 0.4))
            lift = effect if (arm == "T" and triggered) else 0.0
            add(uid, arm, "revenue", base * (1.0 + lift), anchor + dt.timedelta(days=1))
            for name in secondaries:
                secondary_lift = secondary_effect if arm == "T" and triggered else 0.0
                add(
                    uid,
                    arm,
                    name,
                    float(rng.lognormal(0.0, 0.3)) * (1.0 + secondary_lift),
                    anchor + dt.timedelta(days=1),
                )
    return pa.table(
        {
            "user_id": rows["user_id"],
            "group_id": rows["group_id"],
            "ts": pa.array(rows["ts"], type=_UTC_US),
            "event": rows["event"],
            "value": rows["value"],
            "experiment_id": rows["experiment_id"],
            "region": ["north" if uid[-1] in "02468" else "south" for uid in rows["user_id"]],
        }
    )


_UNSET = object()


def _evidence(cutoff: dt.datetime, watermark: object = _UNSET) -> SourceSnapshotEvidence:
    """Pinned evidence; the watermark defaults to the cutoff and ``None`` omits it."""
    feeds: dict[str, dt.datetime | None] = {}
    if watermark is _UNSET:
        feeds["events"] = cutoff
    elif isinstance(watermark, dt.datetime):
        feeds["events"] = watermark
    return SourceSnapshotEvidence(observation_cutoff_ts=cutoff, complete_through_by_feed=feeds)


_CERTIFIED = _evidence(dt.datetime(2024, 1, 31, tzinfo=dt.UTC))
_AS_OF = dt.date(2024, 1, 10)
_LATER = dt.date(2024, 1, 14)


def _connected(path, connection, **options):
    """Bind definitions to a connection; the connected resolver owns a closeable source."""
    experiment, resolver = from_definitions(path, connection, experiment_name="exp", **options)
    assert isinstance(resolver, AbstractContextManager)
    return experiment, resolver


def _pair(
    tmp_path, table, *, plan=_PRIMARY_PLAN, evidence=_CERTIFIED, connection=None, **definitions
):
    path = tmp_path / "defs.yml"
    path.write_text(_definitions(plan=plan, **definitions))
    con = ibis.duckdb.connect() if connection is None else connection
    con.create_table("events", obj=table)
    experiment, resolver = _connected(path, con, source_snapshot_evidence=evidence)
    return experiment, resolver, path


def _analysis(tmp_path, table, **options):
    """A definitions binding over *table*, and the definitions path it was read from."""
    experiment, resolver, path = _pair(tmp_path, table, **options)
    return (experiment, resolver), path


def _by_population(rows):
    return {(row.analysis_population, row.metric): row for row in rows}


_POPULATION_VIEWS = (
    Scope(view="run", grain="total"),
    Scope(view="run", grain="total", population="triggered"),
)


def _both_populations(experiment, resolver, **options) -> Any:
    """The assigned and triggered whole-window rows of one run, as one collection."""
    result = run(experiment.with_views(*_POPULATION_VIEWS), resolver, **options)
    assigned, triggered = (result.readout(scope) for scope in _POPULATION_VIEWS)
    return cast("LiftEstimates", assigned).concat(cast("LiftEstimates", triggered))


def _sequential_rows(result):
    """Each run cell's retained sequential row, as its portable JSON fields."""
    from increment.declarations import thaw

    return [thaw(cell.components["sequential"][1].fields) for cell in result.cells]


def _assert_selected_registered_family_capture(tmp_path):
    connection = ibis.duckdb.connect()
    try:
        path = tmp_path / "family-definitions.yml"
        path.write_text(_definitions(plan=_FAMILY_PLAN, secondaries=("aux_a", "aux_b")))
        connection.create_table("events", obj=_events(secondaries=("aux_a", "aux_b")))
        experiment, resolver = _connected(
            path, connection, source_snapshot_evidence=_CERTIFIED, store="none"
        )
        with resolver:
            assert run(experiment, resolver, metrics=[]).cells == ()
            scopes = (
                Scope(view="run", grain="total", population="assigned"),
                Scope(view="run", grain="total", population="triggered"),
            )
            result = run(
                replace(experiment, scopes=scopes),
                resolver,
                metrics=["aux_a"],
                capture=SequentialCaptureRequest(finalized=True, as_of=_LATER),
            )
            assert {cell.identity.key.metric for cell in result.cells} == {"aux_a"}
            assert {cell.identity.key.analysis_population for cell in result.cells} == {
                "assigned",
                "triggered",
            }
            capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
            connection.raw_sql("DROP TABLE events")
            reversed_result = run(
                replace(experiment, scopes=(scopes[1], scopes[0])),
                resolver,
                metrics=["aux_a"],
            )
            assert {cell.identity.key.analysis_population for cell in reversed_result.cells} == {
                "assigned",
                "triggered",
            }
    finally:
        connection.disconnect()


def test_registered_segmented_asof_refuses_before_shared_capture(tmp_path):
    from increment.evidence import BreakoutIdentity

    connection = ibis.duckdb.connect()
    path = tmp_path / "defs.yml"
    path.write_text(_definitions(plan=_PRIMARY_PLAN, breakouts=True))
    connection.create_table("events", obj=_events(n_per_arm=10))
    experiment, resolver = from_definitions(
        path,
        connection,
        experiment_name="exp",
        source_snapshot_evidence=_CERTIFIED,
        store="none",
    )
    run_scope = Scope(view="run", grain="total")
    run_experiment = replace(
        experiment,
        scopes=(run_scope,),
        operations={run_scope.hash(): "run"},
    )
    try:
        initial = run(
            run_experiment,
            resolver,
            capture=SequentialCaptureRequest(finalized=True, as_of=_AS_OF),
        )
        initial_state = initial.cells[0].components["sequential"][1]
        assert initial_state is not None
        initial_prefix = initial_state.fields["sequential_result"]["checkpoint"]["prefix_id"]
        scopes = (
            run_scope,
            Scope(
                view="asof",
                grain="asof",
                segment=(BreakoutIdentity("region", None),),
            ),
        )
        experiment = replace(
            experiment,
            scopes=scopes,
            operations={
                scopes[0].hash(): "run",
                scopes[1].hash(): "run_asof_lift",
            },
        )
        with pytest.raises(CodedError) as refused:
            run(
                experiment,
                resolver,
                capture=SequentialCaptureRequest(finalized=True, as_of=_LATER),
            )
        assert refused.value.code == "sequential.route.unsupported"
        after = run(run_experiment, resolver)
        after_state = after.cells[0].components["sequential"][1]
        assert after_state is not None
        assert after_state.fields["sequential_result"]["checkpoint"]["prefix_id"] == initial_prefix
    finally:
        connection.disconnect()


def _assert_empty_sequential_compliance(tmp_path):
    compliance_path = tmp_path / "compliance"
    compliance_path.mkdir()
    connection = ibis.duckdb.connect()
    (experiment, resolver), _ = _analysis(
        compliance_path,
        _events(uptake=True),
        plan=_COMPLIANCE_PLAN,
        connection=connection,
        encouragement=True,
    )
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        with pytest.raises(CodedError) as refused:
            _both_populations(experiment, resolver, metrics=[], estimands=("late",))
        assert refused.value.code == "sequential.route.unsupported"
        default_rows = _both_populations(experiment, resolver, metrics=[])
        assert {row.estimand for row in default_rows} == {"compliance"}
        rows = _both_populations(experiment, resolver, metrics=[], estimands=("compliance",))
        assert {row.estimand for row in rows} == {"compliance"}
        # Separate run calls may have distinct snapshot IDs; compare all readout data and state.
        assert [row.model_dump(exclude={"source_snapshot_id"}) for row in default_rows] == [
            row.model_dump(exclude={"source_snapshot_id"}) for row in rows
        ]
        assert default_rows.sequential_snapshot == rows.sequential_snapshot
    finally:
        resolver.close()
        connection.disconnect()


def _assert_triggered_only_store_roundtrip(tmp_path, path, connection):
    from increment.store import ParquetResultsStore

    experiment, resolver = _connected(
        path, connection, source_snapshot_evidence=_CERTIFIED, store="none"
    )
    with resolver:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        scoped = replace(
            experiment,
            scopes=(Scope(view="run", grain="total", population="triggered"),),
        )
        result = run(scoped, resolver)
        assert {cell.identity.key.analysis_population for cell in result.cells} == {"triggered"}
        metadata = result.run.metadata
        assert metadata is not None and metadata.partial
        assert {(record.source_snapshot_id, record.cell) for record in metadata.cells} == {
            (cell.identity.source_snapshot_id, cell.identity.key) for cell in result.cells
        }
        store = ParquetResultsStore(tmp_path / "triggered-run")
        store.put(result.run)
        restored_run = store.open_run(
            result.run.experiment_id, result.run.analysis_spec_hash, result.run.run_id
        ).run
        assert restored_run.cells == result.run.cells


def _assert_combined_compliance_family_requests(tmp_path):
    connection = ibis.duckdb.connect()
    try:
        data_dir = tmp_path / "combined-compliance"
        data_dir.mkdir()
        path = data_dir / "definitions.yml"
        path.write_text(
            _definitions(plan=_COMPLIANCE_PLAN, encouragement=True),
            encoding="utf-8",
        )
        connection.create_table("events", obj=_events(uptake=True))
        experiment, resolver = from_definitions(
            path,
            connection,
            experiment_name="exp",
            source_snapshot_evidence=_CERTIFIED,
            store="none",
        )
        scopes = (
            Scope(view="run", grain="total", population="assigned"),
            Scope(view="run", grain="total", population="triggered"),
        )
        combined = replace(experiment, scopes=scopes)
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        mixed = run(
            combined,
            resolver,
            metrics=["revenue"],
            estimands=["itt", "compliance"],
        )
        assert {
            (cell.identity.key.analysis_population, cell.identity.key.metric)
            for cell in mixed.cells
        } >= {("assigned", "revenue"), ("triggered", "revenue")}
        captured = run(
            combined,
            resolver,
            estimands=["compliance"],
            capture=SequentialCaptureRequest(finalized=True, as_of=_LATER),
        )
        assert {
            (cell.identity.key.analysis_population, cell.identity.key.estimand)
            for cell in captured.cells
        } == {("assigned", "compliance"), ("triggered", "compliance")}
    finally:
        connection.disconnect()


@pytest.mark.integration
def test_registered_combined_compliance_replays_after_event_source_removal(tmp_path):
    path = tmp_path / "combined-compliance-replay.yml"
    path.write_text(_definitions(plan=_COMPLIANCE_PLAN, encouragement=True), encoding="utf-8")
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=_events(uptake=True))
    try:
        experiment, resolver = from_definitions(
            path,
            connection,
            experiment_name="exp",
            source_snapshot_evidence=_CERTIFIED,
            store="none",
        )
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        connection.drop_table("events")
        scopes = (
            Scope(view="run", grain="total", population="assigned"),
            Scope(view="run", grain="total", population="triggered"),
        )
        combined = replace(experiment, scopes=scopes)

        result = run(combined, resolver, estimands=["compliance"])

        assert {
            (cell.identity.key.analysis_population, cell.identity.key.estimand)
            for cell in result.cells
        } == {("assigned", "compliance"), ("triggered", "compliance")}
    finally:
        connection.disconnect()


@pytest.mark.integration
def test_triggered_daily_family_refuses_before_assigned_warehouse_read(tmp_path):
    path = tmp_path / "triggered-daily-family.yml"
    path.write_text(
        _definitions(plan="      primary: revenue\n      alternative: two-sided\n"),
        encoding="utf-8",
    )
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=_events())
    try:
        experiment, resolver = from_definitions(
            path,
            connection,
            experiment_name="exp",
            store="none",
        )
        connection.drop_table("events")
        scopes = (
            Scope(view="run", grain="total", population="assigned"),
            Scope(view="daily", grain="daily", population="triggered"),
        )
        operations = {scopes[0].hash(): "run", scopes[1].hash(): "run_daily_lift"}
        combined = replace(experiment, scopes=scopes, operations=operations)

        with pytest.raises(CodedError) as raised:
            run(combined, resolver, metrics=["revenue"])

        assert raised.value.code == "source.native.trigger_evidence_required"
    finally:
        connection.disconnect()


# Native capture, moments export and persisted replay must retain both populations.
@pytest.mark.slow
def test_trigger_declared_capture_reports_both_populations(tmp_path):
    connection = ibis.duckdb.connect()
    (experiment, resolver), path = _analysis(tmp_path, _events(), connection=connection)
    moments_closer = None
    try:
        _assert_empty_sequential_compliance(tmp_path)
        _assert_selected_registered_family_capture(tmp_path)
        _assert_combined_compliance_family_requests(tmp_path)
        snapshot = capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        assert snapshot.triggered is not None
        assert snapshot.triggered.population == "triggered"
        assert 0 < len(snapshot.triggered.records) < len(snapshot.records)
        assert snapshot.triggered.registration_id != snapshot.registration_id
        rows = _both_populations(experiment, resolver)
        by = _by_population(rows)
        assigned, triggered = by["assigned", "revenue"], by["triggered", "revenue"]
        for row in (assigned, triggered):
            assert row.failure_code is None
            assert row.sampling_available is True
            assert row.decision_scope_complete is True
            result = row.require_sequential_result()
            assert result.checkpoint.population == row.analysis_population
        assert triggered.require_lift().value > assigned.require_lift().value
        assert rows.metadata.scope.decision_complete("assigned") is True
        assert rows.metadata.scope.decision_complete("triggered") is True
        assert {family.analysis_population for family in rows.metadata.scope.families} == {
            "assigned",
            "triggered",
        }
        assert {
            row.analysis_population
            for row in view_readout(experiment, resolver, population="triggered")
        } == {"triggered"}
        restored = ReadoutResults.model_validate_json(rows.model_dump_json())
        assert restored.sequential_snapshot == snapshot
        assert [(row.analysis_population, row.require_sequential_result()) for row in restored] == [
            (row.analysis_population, row.require_sequential_result()) for row in rows
        ]
        history = view_readout(
            experiment, resolver, Scope(view="asof", grain="asof", population="triggered")
        )
        assert [(row.analysis_population, row.ds) for row in history] == [("triggered", _LATER)]
        assert history[0].sequential_result == triggered.require_sequential_result()

        _assert_triggered_only_store_roundtrip(tmp_path, path, connection)
        moments_path = tmp_path / "triggered-sequential.parquet"
        export_moments(experiment, resolver, moments_path)
        moments_experiment, moments_resolver = from_moments(
            pq.read_table(moments_path).to_pylist(),
            metrics={"revenue": "mean"},
            control="C",
        )
        moments_closer = moments_resolver
        moments_scoped = replace(
            moments_experiment,
            scopes=(
                Scope(view="run", grain="total", population="triggered"),
                Scope(
                    view="asof",
                    grain="asof",
                    population="triggered",
                    completion_policy=True,
                ),
            ),
        )
        combined = run(moments_scoped, moments_resolver)
        assert {getattr(cell.identity.scope, "view", None) for cell in combined.cells} == {
            "run",
            "asof",
        }
        assert {cell.identity.key.analysis_population for cell in combined.cells} == {"triggered"}
    finally:
        if moments_closer is not None:
            moments_closer.close()
        resolver.close()
        connection.disconnect()


def _triggered_source_first_metadata(metadata_payload: dict[str, Any]) -> dict[str, Any]:
    triggered_first = json.loads(json.dumps(metadata_payload))
    scope = triggered_first["scope"]
    source_id = next(iter(scope["by_source"]))
    source = scope["by_source"].pop(source_id)
    legacy_source_ids = {"triggered": "a-triggered", "assigned": "z-assigned"}
    for population, legacy_source_id in legacy_source_ids.items():
        source_cells = [
            cell for cell in scope["cells"] if cell["analysis_population"] == population
        ]
        source_decision_cells = [
            cell for cell in scope["decision_cells"] if cell["analysis_population"] == population
        ]
        scope["by_source"][legacy_source_id] = {
            **source,
            "source_snapshot_id": legacy_source_id,
            "cells": source_cells,
            "decision_cells": source_decision_cells,
            "rosters": [
                roster
                for roster in source["rosters"]
                if roster["analysis_population"] == population
            ],
            "integrity": [
                result
                for result in source["integrity"]
                if result["analysis_population"] == population
            ],
            "decision_complete_by_population": {
                population: source["decision_complete_by_population"][population]
            },
        }
        for record in triggered_first["cells"]:
            if record["cell"]["analysis_population"] == population:
                record["source_snapshot_id"] = legacy_source_id
    scope["families"] = []
    triggered_first["cells"] = sorted(
        triggered_first["cells"],
        key=lambda record: (
            record["source_snapshot_id"],
            json.dumps(record["cell"], sort_keys=True, separators=(",", ":")),
        ),
    )
    return triggered_first


@pytest.mark.integration
@pytest.mark.slow
def test_triggered_sequential_metadata_orders_mixed_direction_cells(tmp_path):
    plan = (
        "      primary: revenue\n"
        "      guardrails: [{metric: guard, margin: 0.1}]\n"
        "      alternative: two-sided\n"
        "      inference:\n"
        "        kind: asymptotic_mean\n"
    )
    definitions = _definitions(plan=plan, secondaries=("guard",)).replace(
        "  - name: guard\n    type: mean\n",
        "  - name: guard\n    type: mean\n    preferred_direction: decrease\n",
        1,
    )
    path = tmp_path / "defs.yml"
    path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(secondaries=("guard",)))
    experiment, resolver = from_definitions(
        path, con, experiment_name="exp", source_snapshot_evidence=_CERTIFIED
    )
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        rows = _both_populations(experiment, resolver)
        assert rows.metadata is not None
        legacy_metadata = rows.metadata
        assert legacy_metadata is not None
        cells = rows.metadata.cells
        assert {record.cell.alternative for record in cells if record.cell.metric == "revenue"} == {
            "two-sided"
        }
        assert {record.cell.alternative for record in cells if record.cell.metric == "guard"} == {
            "less"
        }
        populations = [record.cell.analysis_population for record in cells]
        assert populations == sorted(populations, key={"assigned": 0, "triggered": 1}.__getitem__)
        legacy = json.loads(rows.model_dump_json())
        legacy_rows = rows
        legacy_metadata_payload = legacy["metadata"]
        legacy_records = sorted(
            legacy_metadata_payload["cells"],
            key=lambda record: (
                record["source_snapshot_id"],
                json.dumps(record["cell"], sort_keys=True, separators=(",", ":")),
            ),
        )
        legacy_metadata_payload["cells"] = legacy_records
        restored = ReadoutResults.model_validate_json(json.dumps(legacy))
        restored_metadata = restored.metadata
        assert restored_metadata is not None
        assert restored_metadata.cells == legacy_metadata.cells

        triggered_first = _triggered_source_first_metadata(legacy_metadata_payload)
        assert [record["source_snapshot_id"] for record in triggered_first["cells"]] == [
            "a-triggered",
            "a-triggered",
            "z-assigned",
            "z-assigned",
        ]
        assert [
            (record["cell"]["metric"], record["cell"]["analysis_population"])
            for record in triggered_first["cells"]
        ] == [
            ("guard", "triggered"),
            ("revenue", "triggered"),
            ("guard", "assigned"),
            ("revenue", "assigned"),
        ]
        restored_old_order = type(legacy_metadata).from_saved_payload(triggered_first)
        assert [
            (record.cell.metric, record.cell.analysis_population)
            for record in restored_old_order.cells
        ] == [
            ("guard", "assigned"),
            ("revenue", "assigned"),
            ("guard", "triggered"),
            ("revenue", "triggered"),
        ]
        assert [record.source_snapshot_id for record in restored_old_order.cells] == [
            "z-assigned",
            "z-assigned",
            "a-triggered",
            "a-triggered",
        ]
        malformed = json.loads(rows.model_dump_json())
        malformed["metadata"]["cells"].reverse()
        with pytest.raises(CodedError) as raised:
            ReadoutResults.model_validate_json(json.dumps(malformed))
        assert raised.value.code == "readout.serialization.noncanonical_order"
        from increment.result_envelope import AnalysisEnvelope

        envelope = AnalysisEnvelope(tuple(legacy_rows.as_cells()), legacy_metadata)
        legacy_envelope = envelope.to_payload(include_capture_metadata=True)
        legacy_envelope["metadata"]["cells"] = legacy_records
        restored_payload = AnalysisEnvelope.from_payload(legacy_envelope)
        restored_payload_metadata = restored_payload.metadata
        assert restored_payload_metadata is not None
        assert restored_payload_metadata.cells == legacy_metadata.cells
        arrow = envelope.to_arrow()
        arrow_metadata = dict(arrow.schema.metadata or {})
        arrow_metadata[b"increment.readout_metadata"] = json.dumps(
            legacy_envelope["metadata"]
        ).encode("utf-8")
        legacy_arrow = arrow.replace_schema_metadata(arrow_metadata)
        restored_arrow = AnalysisEnvelope.from_arrow(legacy_arrow)
        restored_arrow_metadata = restored_arrow.metadata
        assert restored_arrow_metadata is not None
        assert restored_arrow_metadata.cells == legacy_metadata.cells
    finally:
        cast(Any, resolver).close()


def test_triggered_asof_replays_from_exported_moments(tmp_path):
    import pyarrow.parquet as pq

    (analysis_experiment, analysis_resolver), _ = _analysis(tmp_path, _events())
    path = tmp_path / "triggered.parquet"
    try:
        capture_sequential(analysis_experiment, analysis_resolver, finalized=True, as_of=_LATER)
        export_moments(analysis_experiment, analysis_resolver, path)
        replay_experiment, replay_resolver = from_moments(
            pq.read_table(path).to_pylist(), metrics={"revenue": "mean"}, control="C"
        )
        history = view_readout(
            replay_experiment,
            replay_resolver,
            Scope(view="asof", grain="asof", population="triggered"),
        )
        assert history
        assert {row.analysis_population for row in history} == {"triggered"}
        assert {row.ds for row in history} == {_LATER}
        from dataclasses import replace

        from increment import run, runnable

        experiment, resolver = from_moments(
            pq.read_table(path).to_pylist(),
            metrics={"revenue": "mean"},
            control="C",
        )
        scope = Scope(view="asof", grain="asof", population="triggered")
        requested = replace(experiment, scopes=(scope,))
        assert runnable(requested, "revenue")["randomized.unadjusted"].static_refusals == ()
        resolved_history = run(requested, resolver)
        assert resolved_history.cells
        assert {cell.identity.key.analysis_population for cell in resolved_history.cells} == {
            "triggered"
        }
        assert {cell.identity.key.ds for cell in resolved_history.cells} == {_LATER}

    finally:
        analysis_resolver.close()


def test_triggered_registration_is_derived_before_data(tmp_path):
    (experiment, resolver), _ = _analysis(tmp_path, _events(n_per_arm=10))
    try:
        spec = experiment_definition(experiment, resolver).plan.inference
        assert spec is not None and spec.registration is not None
        triggered = spec.triggered_registration
        assert triggered is not None
        assert triggered.population == "triggered"
        assert triggered.definitions_id != spec.registration.definitions_id
        assert triggered.models == spec.registration.models
        assert triggered.roster == spec.registration.roster
        assert triggered.reveal.filtration_id != spec.registration.reveal.filtration_id
        rhos = {cast(Any, model).rho for model in (*triggered.models, *spec.registration.models)}
        assert len(rhos) == 1
        with pytest.raises(CodedError) as declared:
            InferenceSpec(kind="asymptotic_mean", triggered_registration=triggered)
        assert declared.value.code == "sequential.registration.invalid"
        foreign = triggered.model_copy(update={"q": Fraction(1, 5)})
        with pytest.raises(CodedError) as mismatched:
            InferenceSpec(
                kind="asymptotic_mean",
                registration=spec.registration,
                triggered_registration=foreign,
            )
        assert mismatched.value.code == "sequential.registration.invalid"
    finally:
        resolver.close()


@pytest.mark.slow
@pytest.mark.parametrize("late_anchor", ["inside_prefix", "after_prefix"])
def test_triggered_prefix_extends_and_a_late_trigger_refuses(tmp_path, late_anchor):
    table = _events()
    experiment, resolver, path = _pair(tmp_path, table)
    with resolver:
        first = capture_sequential(experiment, resolver, finalized=True, as_of=_AS_OF)
        later = capture_sequential(
            experiment, resolver, finalized=True, as_of=_LATER, previous=first
        )
        assert later.triggered is not None and first.triggered is not None
        assert len(later.triggered.records) > len(first.triggered.records)
        assert later.triggered.records[: len(first.triggered.records)] == first.triggered.records
        later.verify_parent(first)
        assert later.triggered.parent_id == first.triggered.prefix_id
        assert later.records[: len(first.records)] == first.records
    # A trigger that arrives after the first look but whose window had already
    # closed by then belongs to a reported look, whether it sorts between the
    # retained records (an hour after its assignment) or after every one of them
    # (late on the last day whose window closes at the first look).
    rows = table.to_pylist()
    triggered_units = {row["user_id"] for row in rows if row["event"] == "saw_surface"}
    exposure = next(
        row for row in rows if row["event"] == "enrolled" and row["user_id"] not in triggered_units
    )
    anchor = (
        exposure["ts"] + dt.timedelta(hours=1)
        if late_anchor == "inside_prefix"
        else dt.datetime(2024, 1, 8, 23, tzinfo=dt.UTC)
    )
    assert anchor >= exposure["ts"]
    late = pa.table(
        {
            "user_id": [exposure["user_id"]],
            "group_id": [exposure["group_id"]],
            "ts": pa.array([anchor], type=_UTC_US),
            "event": ["saw_surface"],
            "value": [0.0],
            "experiment_id": ["exp"],
            "region": [exposure["region"]],
        }
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.concat_tables([table, late]))
    backfilled_experiment, backfilled = _connected(path, con, source_snapshot_evidence=_CERTIFIED)
    with backfilled, pytest.raises(CapabilityError) as raised:
        capture_sequential(
            backfilled_experiment, backfilled, finalized=True, as_of=_LATER, previous=first
        )
    assert raised.value.code == "sequential.continuation.rewrite"


@pytest.mark.parametrize(
    ("evidence", "feed"),
    [
        (_evidence(dt.datetime(2024, 1, 10, 12, tzinfo=dt.UTC)), "events"),
        (_evidence(dt.datetime(2024, 1, 31, tzinfo=dt.UTC), None), "events"),
        (
            _evidence(
                dt.datetime(2024, 1, 31, tzinfo=dt.UTC), dt.datetime(2024, 1, 9, tzinfo=dt.UTC)
            ),
            "events",
        ),
    ],
    ids=["cutoff_during_as_of", "no_watermark", "watermark_before_as_of"],
)
def test_capture_refuses_feeds_not_certified_through_as_of(tmp_path, evidence, feed):
    experiment, resolver, _ = _pair(tmp_path, _events(n_per_arm=20), evidence=evidence)
    with resolver, pytest.raises(CapabilityError) as raised:
        capture_sequential(experiment, resolver, finalized=True, as_of=_AS_OF)
    assert raised.value.code == "sequential.source.invalid"
    assert_refusal(
        "triggered-sequential-certification", "native-feed", raised.value, stage="capture"
    )
    assert raised.value.context["feed"] == feed
    assert raised.value.context["as_of"] == _AS_OF.isoformat()
    certified = raised.value.context["certified_day"]
    assert certified is None or dt.date.fromisoformat(str(certified)) < _AS_OF
    for copied in (pickle.loads(pickle.dumps(raised.value)), copy.deepcopy(raised.value)):
        assert copied.code == "sequential.source.invalid"
        assert copied.context == raised.value.context


def test_capture_refuses_a_sql_declared_trigger_and_missing_evidence(tmp_path):
    experiment, resolver, _ = _pair(tmp_path, _events(n_per_arm=20), trigger_sql=True)
    with resolver, pytest.raises(CapabilityError) as raised:
        capture_sequential(experiment, resolver, finalized=True, as_of=_AS_OF)
    assert raised.value.code == "sequential.source.invalid"
    assert_refusal(
        "triggered-sequential-certification", "native-sql", raised.value, stage="capture"
    )
    assert raised.value.context["feed"] == "saw_surface"
    path = tmp_path / "no-evidence.yml"
    path.write_text(_definitions(plan=_PRIMARY_PLAN))
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20))
    unpinned_experiment, unpinned = _connected(path, con)
    with unpinned, pytest.raises(CapabilityError) as raised:
        capture_sequential(unpinned_experiment, unpinned, finalized=True, as_of=_AS_OF)
    assert raised.value.code == "source.native.trigger_evidence_required"
    assert_refusal(
        "triggered-sequential-certification", "native-unpinned", raised.value, stage="capture"
    )


def test_artifact_capture_refuses_an_uncertified_trigger_feed(tmp_path):
    connection = ibis.duckdb.connect()
    experiment, resolver, path = _pair(
        tmp_path,
        _events(n_per_arm=20),
        evidence=_evidence(dt.datetime(2024, 1, 31, tzinfo=dt.UTC), None),
        connection=connection,
    )
    with resolver:
        store = WarehouseArtifactStore(connection, schema_name="uncertified")
        defs = load(path)
        context = artifact_context(defs, defs.experiments[0], "error")
        requests = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "assignment_counts", "trigger_measure_stats"}
        ]
        reference = publish_unit_day_artifact(experiment, resolver, store, extensions=requests)
        adopted_experiment, adopted = from_unit_day_artifact(
            store, reference, expected_context=context
        )
        with adopted, pytest.raises(CapabilityError) as raised:
            capture_sequential(adopted_experiment, adopted, finalized=True, as_of=_AS_OF)
        assert raised.value.code == "sequential.source.invalid"
        assert raised.value.context["certified_day"] is None
        assert_refusal(
            "triggered-sequential-certification", "artifact-feed", raised.value, stage="capture"
        )


@pytest.mark.slow
def test_assigned_only_process_continues_assigned_only_and_refuses_backfill(tmp_path):
    (experiment, resolver), _ = _analysis(tmp_path, _events())
    try:
        committed = capture_sequential(experiment, resolver, finalized=True, as_of=_AS_OF)
        assert committed.triggered is not None
        legacy = SequentialSnapshot.model_validate(
            committed.model_copy(update={"triggered": None}).model_dump()
        )
        continued = capture_sequential(
            experiment, resolver, finalized=True, as_of=_LATER, previous=legacy
        )
        assert continued.triggered is None
        assert len(continued.records) >= len(legacy.records)
        rows = _both_populations(experiment, resolver)
        by = _by_population(rows)
        assert by["assigned", "revenue"].sequential_result is not None
        placeholder = by["triggered", "revenue"]
        assert placeholder.failure_code == "readout.cell.unsupported_request"
        assert placeholder.failure_context is not None
        assert placeholder.failure_context["reason"] == "triggered_chain_uncommitted"
        assert placeholder.lift is None and placeholder.sampling_available is False
        assert rows.metadata.scope.decision_complete("assigned") is True
        assert rows.metadata.scope.decision_complete("triggered") is False
        assert by["assigned", "revenue"].decision_scope_complete is True
        restored = ReadoutResults.model_validate_json(rows.model_dump_json())
        assert [(row.analysis_population, row.failure_code) for row in restored] == [
            ("assigned", None),
            ("triggered", "readout.cell.unsupported_request"),
        ]
        history = view_readout(
            experiment, resolver, Scope(view="asof", grain="asof", population="triggered")
        )
        assert [(row.ds, row.failure_context["reason"]) for row in history] == [
            (_LATER, "triggered_chain_uncommitted")
        ]
        # A committed process cannot be linked onto the assigned-only one the
        # source retained, and an assigned-only parent never acquires the chain.
        with pytest.raises(CapabilityError) as dropped:
            sequential_snapshot(experiment, resolver, previous=committed)
        assert dropped.value.code == "sequential.continuation.rewrite"
        with pytest.raises(CapabilityError) as backfill:
            committed.verify_parent(legacy)
        assert backfill.value.code == "sequential.continuation.legacy"
    finally:
        resolver.close()
    fresh_experiment, fresh, _ = _pair(tmp_path, _events())
    with fresh:
        capture_sequential(fresh_experiment, fresh, finalized=True, as_of=_AS_OF)
        with pytest.raises(CapabilityError) as linked:
            sequential_snapshot(fresh_experiment, fresh, previous=legacy)
        assert linked.value.code == "sequential.continuation.legacy"


@pytest.mark.parametrize(
    "ingress",
    [
        "from_definitions",
        pytest.param("from_unit_day_artifact", marks=pytest.mark.slow),
        "from_moments",
    ],
)
def test_triggered_uptake_cells_stay_unavailable(tmp_path, ingress):
    connection = ibis.duckdb.connect()
    native_experiment, native, path = _pair(
        tmp_path,
        _events(uptake=True),
        plan=_COMPLIANCE_PLAN,
        encouragement=True,
        connection=connection,
    )
    experiment, resolver = native_experiment, native
    try:
        spec = experiment_definition(experiment, resolver).plan.inference
        assert spec is not None and spec.registration is not None
        assert any(model.observable == "uptake" for model in spec.registration.models)
        triggered = spec.triggered_registration
        assert triggered is not None
        assert all(model.observable == "outcome" for model in triggered.models)
        assert all(cell.estimand != "compliance" for cell in triggered.roster)
        snapshot = capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        assert snapshot.triggered is not None
        if ingress == "from_unit_day_artifact":
            store, reference, context = _publish_with_trigger_evidence(
                (native_experiment, native),
                path,
                connection=connection,
                schema="uptake_availability",
                kinds={
                    "trigger_population",
                    "assignment_counts",
                    "trigger_measure_stats",
                    "encouragement_uptake",
                },
            )
            experiment, resolver = from_unit_day_artifact(
                store, reference, expected_context=context
            )
            capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        elif ingress == "from_moments":
            destination = tmp_path / "uptake.parquet"
            export_moments(native_experiment, native, destination)
            experiment, resolver = from_moments(
                pq.read_table(destination).to_pylist(),
                metrics={"revenue": "mean"},
                design=experiment_definition(native_experiment, native).resolved_design(),
            )
        rows = cast(
            "LiftEstimates",
            _both_populations(experiment, resolver, estimands=("itt", "compliance")),
        )
        cells = {
            (row.analysis_population, row.metric, row.estimand): row
            for row in rows
            if row.method_role == "decision"
        }
        assert cells["triggered", "revenue", "itt"].sequential_result is not None
        uptake = cells["triggered", "uptake", "compliance"]
        assert uptake.failure_code == "readout.cell.unsupported_request"
        assert uptake.failure_context is not None
        assert uptake.failure_context["reason"] == "triggered_uptake_unsupported"
        declared = assert_expected(
            "triggered-sequential-uptake",
            f"{ingress}-unavailable",
            kind="unavailable",
            code=uptake.failure_code,
            stage="run",
        )
        assert declared.expected.reason == uptake.failure_context["reason"]
        assert cells["assigned", "uptake", "compliance"].sequential_result is not None
        assert rows.metadata is not None
        assert rows.metadata.scope.decision_complete("triggered") is False
    finally:
        cast(Any, resolver).close()
        if resolver is not native:
            native.close()


@pytest.mark.parametrize(
    "ingress",
    [
        "from_definitions",
        pytest.param("from_unit_day_artifact", marks=pytest.mark.slow),
        "from_moments",
    ],
)
def test_assigned_only_state_retains_unavailable_triggered_rows_across_sources(tmp_path, ingress):
    connection = ibis.duckdb.connect()
    native_experiment, native, path = _pair(tmp_path, _events(), connection=connection)
    experiment, resolver = native_experiment, native
    try:
        committed = capture_sequential(native_experiment, native, finalized=True, as_of=_AS_OF)
        legacy = SequentialSnapshot.model_validate(
            committed.model_copy(update={"triggered": None}).model_dump()
        )
        continued = capture_sequential(
            native_experiment, native, finalized=True, as_of=_LATER, previous=legacy
        )
        if ingress == "from_unit_day_artifact":
            store, reference, context = _publish_with_trigger_evidence(
                (native_experiment, native),
                path,
                connection=connection,
                schema="assigned_only",
                kinds={"trigger_population", "assignment_counts", "trigger_measure_stats"},
            )
            experiment, resolver = from_unit_day_artifact(
                store, reference, expected_context=context
            )
            capture_sequential(
                experiment, resolver, finalized=True, as_of=_LATER, previous=continued
            )
        elif ingress == "from_moments":
            destination = tmp_path / "assigned-only.parquet"
            export_moments(native_experiment, native, destination)
            experiment, resolver = from_moments(
                pq.read_table(destination).to_pylist(),
                metrics={"revenue": "mean"},
                design=experiment_definition(native_experiment, native).resolved_design(),
            )
        retained = sequential_snapshot(experiment, resolver)
        assert retained.triggered is None
        assert retained.records == continued.records
        rows = _by_population(_both_populations(experiment, resolver))
        assert rows["assigned", "revenue"].sequential_result is not None
        unavailable = rows["triggered", "revenue"]
        assert unavailable.lift is None and unavailable.sampling_available is False
        assert unavailable.failure_context is not None
        declared = assert_expected(
            "triggered-sequential-continuation",
            f"{ingress}-unavailable",
            kind="unavailable",
            code=unavailable.failure_code,
            stage="run",
        )
        assert declared.expected.reason == unavailable.failure_context["reason"]
        with pytest.raises(CapabilityError) as backfill:
            committed.verify_parent(retained)
        assert_refusal(
            "triggered-sequential-continuation",
            f"{ingress}-backfill",
            backfill.value,
            stage="verify_parent",
        )
    finally:
        cast(Any, resolver).close()
        if resolver is not native:
            native.close()


_COMPLIANCE_PLAN = (
    "      primary: revenue\n      alternative: greater\n"
    "      inference:\n        kind: asymptotic_mean\n"
    "      compliance:\n        alpha: 0.05\n"
)


def _publish_with_trigger_evidence(
    analysis: tuple[Any, Any], path, *, connection, schema: str, kinds: set[str]
):
    store = WarehouseArtifactStore(connection, schema_name=schema)
    defs = load(path)
    context = artifact_context(defs, defs.experiments[0], "error")
    requests = [
        entry.request
        for entry in unit_day_artifact_extension_catalog(context)
        if entry.request.kind in kinds
    ]
    return store, publish_unit_day_artifact(*analysis, store, extensions=requests), context


@pytest.mark.slow
def test_certified_encouragement_artifact_captures_both_chains(tmp_path):
    connection = ibis.duckdb.connect()
    analysis, path = _analysis(
        tmp_path,
        _events(uptake=True),
        plan=_COMPLIANCE_PLAN,
        encouragement=True,
        connection=connection,
    )
    try:
        native = capture_sequential(*analysis, finalized=True, as_of=_LATER)
        store, reference, context = _publish_with_trigger_evidence(
            analysis,
            path,
            connection=connection,
            schema="encouragement_parity",
            kinds={
                "trigger_population",
                "assignment_counts",
                "trigger_measure_stats",
                "encouragement_uptake",
            },
        )
        experiment, resolver = from_unit_day_artifact(store, reference, expected_context=context)
        with resolver:
            artifact = capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
            assert artifact == native
            cells = {
                (row.analysis_population, row.metric, row.estimand): row
                for row in cast(
                    "LiftEstimates",
                    _both_populations(experiment, resolver, estimands=("itt", "compliance")),
                )
                if row.method_role == "decision"
            }
            assert cells["assigned", "uptake", "compliance"].sequential_result is not None
            assert cells["triggered", "revenue", "itt"].sequential_result is not None
            unavailable = cells["triggered", "uptake", "compliance"]
            assert unavailable.failure_context is not None
            assert unavailable.failure_context["reason"] == "triggered_uptake_unsupported"
    finally:
        analysis[1].close()


def test_uncertified_uptake_feed_refuses_on_both_routes(tmp_path):
    # Only the events feed is certified; the separate clicks feed carries no watermark.
    connection = ibis.duckdb.connect()
    analysis, path = _analysis(
        tmp_path,
        _events(uptake=True),
        plan=_COMPLIANCE_PLAN,
        encouragement=True,
        uptake_source=True,
        connection=connection,
    )
    try:
        with pytest.raises(CapabilityError) as native:
            capture_sequential(*analysis, finalized=True, as_of=_LATER)
        assert native.value.code == "sequential.source.invalid"
        assert native.value.context["feed"] == "clicks"
        assert native.value.context["certified_day"] is None
        store, reference, context = _publish_with_trigger_evidence(
            analysis,
            path,
            connection=connection,
            schema="uncertified_uptake",
            kinds={
                "trigger_population",
                "assignment_counts",
                "trigger_measure_stats",
                "encouragement_uptake",
            },
        )
        experiment, resolver = from_unit_day_artifact(store, reference, expected_context=context)
        with resolver:
            with pytest.raises(CapabilityError) as artifact:
                capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
            assert artifact.value.code == "sequential.source.invalid"
            assert artifact.value.context["feed"] == "encouragement_uptake:clicked"
            assert artifact.value.context["certified_day"] is None
    finally:
        analysis[1].close()


def test_assigned_identities_are_unchanged_by_the_triggered_commitment(tmp_path):
    """A process captured before triggered monitoring existed continues assigned-only.

    The assigned observation recipe, and so every assigned record digest, must
    not depend on the derived triggered registration: the same declarations
    compiled without it must yield byte-identical assigned chains, and the
    committed process must continue them.
    """
    from increment.plan import bind_automatic_sequential_plan, compile_decision_plan
    from increment.sequential_source import native_observation_mapping

    table = _events()
    connection = ibis.duckdb.connect()
    (analysis_experiment, analysis_resolver), path = _analysis(
        tmp_path, table, connection=connection
    )
    try:
        defs = load(path)
        experiment = defs.experiments[0]
        context = artifact_context(defs, experiment, "error")
        assert "triggered_registration" not in context.canonical_json
        # The pre-change plan: the same declarations with no triggered derivation bound.
        earlier_path = tmp_path / "earlier.yml"
        earlier_path.write_text(_definitions(plan=_PRIMARY_PLAN))
        earlier_defs = load(earlier_path)
        earlier_experiment = earlier_defs.experiments[0]
        earlier_context = artifact_context(earlier_defs, earlier_experiment, "error")
        assert context.sha256 == earlier_context.sha256
        metrics = [
            metric
            for metric in earlier_defs.metrics
            if metric.name in earlier_experiment.metric_names
        ]
        design = earlier_experiment.resolved_design()
        uncommitted = compile_decision_plan(
            bind_automatic_sequential_plan(
                earlier_experiment.plan,
                metrics,
                design=design,
                source_id=earlier_experiment.name,
                source_mapping=native_observation_mapping(earlier_defs, earlier_experiment),
                pre_period_covariate=earlier_experiment.n_pre_periods > 0,
            ),
            metrics,
            path="warehouse",
            design=design,
        )
        assert getattr(uncommitted.inference, "triggered_registration", None) is None
        earlier = make_binding(
            connection,
            earlier_defs,
            experiment=earlier_experiment,
            plan=uncommitted,
            source_snapshot_evidence=_CERTIFIED,
        )
        try:
            legacy = capture_sequential(*earlier, finalized=True, as_of=dt.date(2024, 1, 6))
        finally:
            earlier[1].close()
        assert legacy.triggered is None
        committed = capture_sequential(
            analysis_experiment, analysis_resolver, finalized=True, as_of=dt.date(2024, 1, 6)
        )
        assert committed.triggered is not None
        assert committed.records == legacy.records
        assert committed.prefix_id == legacy.prefix_id
        continued = capture_sequential(
            analysis_experiment, analysis_resolver, finalized=True, as_of=_LATER, previous=legacy
        )
        assert continued.triggered is None
        assert continued.parent_id == legacy.prefix_id
        assert len(continued.records) > len(legacy.records)
        rows = _both_populations(analysis_experiment, analysis_resolver)
        by = _by_population(rows)
        assert by["assigned", "revenue"].sequential_result is not None
        assert by["triggered", "revenue"].failure_context["reason"] == "triggered_chain_uncommitted"
    finally:
        analysis_resolver.close()


@pytest.mark.integration
def test_assigned_checkpoint_exposes_next_same_metric_triggered_block(tmp_path):
    events = _events(n_per_arm=30, trigger_rate=1.0, effect=0.0).to_pylist()
    assignment = {row["user_id"]: row["ts"] for row in events if row["event"] == "enrolled"}
    for row in events:
        anchor = assignment[row["user_id"]] + dt.timedelta(days=1)
        if row["event"] == "saw_surface":
            row["ts"] = anchor
        elif row["event"] == "revenue":
            row["ts"] = anchor + dt.timedelta(hours=1)
    experiment, resolver, _ = _pair(tmp_path, pa.Table.from_pylist(events))
    with resolver:
        early = capture_sequential(
            experiment, resolver, finalized=True, as_of=dt.date(2024, 1, 5), freeze=["revenue"]
        )
        assert early.triggered is not None
        assert (
            early.registration.reveal.filtration_id
            != early.triggered.registration.reveal.filtration_id
        )
        assert len(early.records) > len(early.triggered.records)
        later = capture_sequential(
            experiment, resolver, finalized=True, as_of=dt.date(2024, 1, 6), previous=early
        )
        assert later.triggered is not None
        for arm in ("C", "T"):
            assert early.arm("revenue", arm) == later.triggered.arm("revenue", arm)
            assert early.arm("revenue", arm).n > early.triggered.arm("revenue", arm).n
        frozen = later.triggered.frozen[0]
        assert frozen.control == early.triggered.arm("revenue", "C")
        assert frozen.treatment == early.triggered.arm("revenue", "T")
        triggered = replace(
            experiment, scopes=(Scope(view="run", grain="total", population="triggered"),)
        )
        (row,) = _sequential_rows(run(triggered, resolver))
        assert row["sequential_result"]["checkpoint"] == frozen.model_dump(mode="json")


@pytest.mark.slow
def test_metric_freeze_reaches_the_triggered_chain_once_it_is_ready(tmp_path):
    experiment, resolver, _ = _pair(tmp_path, _events(treatment_trigger_delay=6))
    with resolver:
        early = capture_sequential(
            experiment, resolver, finalized=True, as_of=dt.date(2024, 1, 8), freeze=["revenue"]
        )
        assert early.triggered is not None
        assert len(early.frozen) == 1
        assert early.triggered.frozen == ()
        assert early.triggered.arm("revenue", "T").n == 0
        later = capture_sequential(
            experiment, resolver, finalized=True, as_of=_LATER, previous=early, freeze=["revenue"]
        )
        assert later.triggered is not None
        assert later.frozen == early.frozen
        assert len(later.triggered.frozen) == 1
        assert later.triggered.frozen[0].population == "triggered"
        assert later.triggered.arm("revenue", "T").n > 0
        both = replace(
            experiment,
            scopes=(
                Scope(view="run", grain="total", population="assigned"),
                Scope(view="run", grain="total", population="triggered"),
            ),
        )
        rows = _sequential_rows(run(both, resolver))
        assert {row["analysis_population"] for row in rows} == {"assigned", "triggered"}
        for row in rows:
            assert row["sequential_result"]["checkpoint"]["status"] == "frozen"
        with pytest.raises(CapabilityError) as raised:
            capture_sequential(
                experiment,
                resolver,
                finalized=True,
                as_of=_LATER,
                previous=later,
                freeze=["revenue"],
            )
        assert raised.value.code == "sequential.freeze.invalid"


@pytest.mark.slow
def test_native_and_artifact_capture_identical_chains(tmp_path):
    connection = ibis.duckdb.connect()
    (analysis_experiment, analysis_resolver), path = _analysis(
        tmp_path, _events(), connection=connection
    )
    try:
        native = capture_sequential(
            analysis_experiment, analysis_resolver, finalized=True, as_of=_LATER, freeze=["revenue"]
        )
        assert native.triggered is not None
        assert len(native.frozen) == 1 and len(native.triggered.frozen) == 1
        native_rows = _both_populations(analysis_experiment, analysis_resolver)
        store = WarehouseArtifactStore(connection, schema_name="parity")
        defs = load(path)
        context = artifact_context(defs, defs.experiments[0], "error")
        requests = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "assignment_counts", "trigger_measure_stats"}
        ]
        reference = publish_unit_day_artifact(
            analysis_experiment, analysis_resolver, store, extensions=requests
        )
        adopted_experiment, adopted_resolver = from_unit_day_artifact(
            store, reference, expected_context=context
        )
        with adopted_resolver:
            spec = experiment_definition(adopted_experiment, adopted_resolver).plan.inference
            native_spec = experiment_definition(
                analysis_experiment, analysis_resolver
            ).plan.inference
            assert spec is not None and native_spec is not None
            assert spec.triggered_registration == native_spec.triggered_registration
            artifact = capture_sequential(
                adopted_experiment,
                adopted_resolver,
                finalized=True,
                as_of=_LATER,
                freeze=["revenue"],
            )
            assert artifact == native
            artifact_rows = _both_populations(adopted_experiment, adopted_resolver)
            compared = (
                "analysis_population",
                "lift",
                "sequential_result",
                "discovery",
                "family_threshold",
                "family_guarantee",
                "multiplicity_status",
                "decision_scope_complete",
            )
            assert [row.model_dump(include=set(compared)) for row in artifact_rows] == [
                row.model_dump(include=set(compared)) for row in native_rows
            ]
            replay = ReadoutResults.model_validate_json(artifact_rows.model_dump_json())
            assert replay.sequential_snapshot == native
    finally:
        analysis_resolver.close()


@pytest.mark.slow
def test_final_look_triggered_states_match_the_fixed_horizon_triggered_frame(tmp_path):

    (experiment, resolver), _ = _analysis(tmp_path, _events())
    try:
        snapshot = capture_sequential(
            experiment, resolver, finalized=True, as_of=dt.date(2024, 1, 20)
        )
        assert snapshot.triggered is not None
        frame = request_evidence(
            evidence_provider(bound_source(resolver).triggered_source()),
            UnitRowsRequest(
                next(
                    metric
                    for metric in workflows.metrics(experiment, resolver)
                    if metric.name == "revenue"
                ),
                population="triggered",
            ),
        )
        rows = cast("pa.Table", frame).to_pylist()
        by_arm: dict[str, list[Fraction]] = {}
        for row in rows:
            by_arm.setdefault(str(row["group_id"]), []).append(Fraction(float(row["y"])))
        for state in snapshot.triggered.states:
            values = by_arm[state.group_id]
            assert state.n == len(values)
            mean = sum(values, Fraction(0)) / len(values)
            scatter = sum(((v - mean) ** 2 for v in values), Fraction(0))
            assert state.mean[0] == mean
            assert state.scatter[0][0] == scatter
    finally:
        resolver.close()


def test_secondary_family_is_selected_per_population_with_the_same_guarantee(tmp_path):
    secondaries = ("aux_a", "aux_b")
    (experiment, resolver), _ = _analysis(
        tmp_path, _events(secondaries=secondaries), plan=_FAMILY_PLAN, secondaries=secondaries
    )
    try:
        spec = experiment_definition(experiment, resolver).plan.inference
        assert spec is not None and spec.triggered_registration is not None
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        rows = _both_populations(experiment, resolver)
        families = {
            (family.analysis_population, family.name): family
            for family in rows.metadata.scope.families
        }
        assert (
            families["assigned", "secondary"].family_id
            != families["triggered", "secondary"].family_id
        )
        assigned_family = families["assigned", "secondary"].family
        triggered_family = families["triggered", "secondary"].family
        assert assigned_family is not None and triggered_family is not None
        assert assigned_family == triggered_family
        assert triggered_family.guarantee == "fdr"
        assert triggered_family.correction == "e_bh"
        secondary_rows = [
            row for row in rows if row.metric in secondaries and row.method_role == "decision"
        ]
        assert len(secondary_rows) == 4
        for row in secondary_rows:
            assert row.discovery is not None
            assert row.family_q == pytest.approx(0.1)
            assert row.multiplicity_status == "declared_plan"
            assert row.family_guarantee == "asymptotic_sequential"
        by_population = {
            population: {
                row.metric: row.discovery
                for row in secondary_rows
                if row.analysis_population == population
            }
            for population in ("assigned", "triggered")
        }
        assert set(by_population["assigned"]) == set(by_population["triggered"]) == set(secondaries)
        restored = ReadoutResults.model_validate_json(rows.model_dump_json())
        assert [row.multiplicity_status for row in restored] == [
            row.multiplicity_status for row in rows
        ]
    finally:
        resolver.close()


def _assert_triggered_selection_public_surfaces(rows, selected):
    from increment.tables import estimates_to_readout, readout_table

    restored = ReadoutResults.model_validate_json(rows.model_dump_json())
    qualification = estimates_to_readout([selected])[0]["selection_qualification"]
    assert qualification is not None
    assert repr(qualification) in repr(selected)
    restored_selected = next(
        row
        for row in restored
        if row.analysis_population == "triggered"
        and row.metric == selected.metric
        and row.group_id == selected.group_id
        and row.discovery
    )
    assert restored_selected.discovery is True
    assert restored_selected.family_guarantee == "asymptotic_sequential"
    assert restored_selected.family_procedure == "e_bh"
    assert restored_selected.multiplicity_status == "declared_plan"
    assert restored_selected.family_id == selected.family_id
    assert restored_selected.family_axes == selected.family_axes
    assert restored_selected.family_q == selected.family_q
    assert restored_selected.family_size == selected.family_size
    frame = restored.to_frame(backend="pandas")
    selected_frame = frame[
        (frame["analysis_population"] == "triggered")
        & (frame["metric"] == selected.metric)
        & (frame["group_id"] == selected.group_id)
    ].iloc[0]
    assert selected_frame["family_id"] == selected.family_id
    assert selected_frame["multiplicity_status"] == "declared_plan"
    assert selected_frame["family_procedure"] == "e_bh"
    assert selected_frame["family_q"] == pytest.approx(selected.family_q)
    assert all(axis in selected_frame["family_axes"] for axis in selected.family_axes)
    assert bool(selected_frame["discovery"])
    assert selected_frame["family_guarantee"] == "asymptotic_sequential"
    rendered = readout_table(estimates_to_readout(restored)).gt().as_raw_html()
    assert estimates_to_readout([restored_selected])[0]["selection_qualification"] == qualification
    assert qualification in rendered


def test_triggered_selection_is_qualified_on_public_surfaces(tmp_path):
    secondaries = ("aux_a", "aux_b")
    (experiment, resolver), _ = _analysis(
        tmp_path,
        _events(secondaries=secondaries, secondary_effect=1.0),
        plan=_FAMILY_PLAN,
        secondaries=secondaries,
    )
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        rows = _both_populations(experiment, resolver)
        selected = next(
            row
            for row in rows
            if row.analysis_population == "triggered"
            and row.metric in secondaries
            and row.discovery
        )
        _assert_triggered_selection_public_surfaces(rows, selected)
    finally:
        resolver.close()


def test_filtered_readout_keeps_original_family_qualification(tmp_path):
    from increment.tables import estimates_to_readout, readout_table

    secondaries = ("aux_a", "aux_b")
    (experiment, resolver), _ = _analysis(
        tmp_path,
        _events(secondaries=secondaries, secondary_effect=1.0),
        plan=_FAMILY_PLAN,
        secondaries=secondaries,
    )
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        rows = _both_populations(experiment, resolver)
        selected = next(
            row
            for row in rows
            if row.analysis_population == "triggered"
            and row.metric in secondaries
            and row.discovery
        )
        qualification = estimates_to_readout([selected])[0]["selection_qualification"]
        filtered = rows.filter(
            lambda row: row.analysis_population == "triggered" and row.metric == selected.metric
        )
        assert filtered.metadata.scope.families == rows.metadata.scope.families
        assert filtered.metadata.partial is True
        filtered_selected = next(row for row in filtered if row.discovery)
        assert filtered_selected.family_id == selected.family_id
        assert filtered_selected.family_q == selected.family_q
        filtered_qualification = estimates_to_readout([filtered_selected])[0][
            "selection_qualification"
        ]
        assert filtered_qualification == qualification
        html = readout_table(estimates_to_readout(filtered)).gt().as_raw_html()
        assert "Partial view" in html
        assert qualification in html
    finally:
        resolver.close()


def test_triggered_request_without_a_trigger_refuses_under_a_registered_plan(tmp_path):
    path = tmp_path / "untriggered.yml"
    path.write_text(_definitions(plan=_PRIMARY_PLAN).replace("    trigger: saw_surface\n", ""))
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20))
    experiment, resolver = from_definitions(
        path, con, experiment_name="exp", source_snapshot_evidence=_CERTIFIED
    )
    try:
        spec = experiment_definition(experiment, resolver).plan.inference
        assert spec is not None and spec.triggered_registration is None
        snapshot = capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        assert snapshot.triggered is None
        assert {row.analysis_population for row in view_readout(experiment, resolver)} == {
            "assigned"
        }
        with pytest.raises(CapabilityError) as raised:
            view_readout(experiment, resolver, population="triggered")
        assert raised.value.code == "facade.analysis.trigger_unsupported"
    finally:
        cast(Any, resolver).close()


def test_breakout_route_keeps_refusing_triggered_sequential_segments(tmp_path):
    (experiment, resolver), _ = _analysis(tmp_path, _events(n_per_arm=20), breakouts=True)
    try:
        capture_sequential(experiment, resolver, finalized=True, as_of=_LATER)
        with pytest.raises(CapabilityError) as raised:
            view_readout(
                experiment, resolver, Scope(view="breakout", grain="total", population="triggered")
            )
        assert raised.value.code == "sequential.route.unsupported"
    finally:
        resolver.close()


pytestmark = pytest.mark.integration
