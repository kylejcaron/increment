"""Triggered analysis: declaring a trigger, narrowing the population, and
reporting both effects.

A feature that fires for part of the assigned population has a true effect
and a diluted one. These tests pin that both are reported, and that a
degenerate trigger is refused rather than analyzed.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import ibis
import numpy as np
import pyarrow as pa
import pytest

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import (
    CapabilityError,
    DefinitionError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.estimation.results import LiftEstimate
from increment.semantics import load
from tests.warning_codes import warning_codes


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


def _defs_yaml(
    trigger: str | None = "saw_surface",
    *,
    allocation_scheme: str | None = None,
    end: str | None = None,
    observation_end: str | None = None,
) -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    scheme_line = (
        f"    allocation: {{C: 0.5, T: 0.5}}\n    allocation_scheme: {allocation_scheme}\n"
        if allocation_scheme
        else ""
    )
    end_line = f"    end: {end}\n" if end else ""
    observation_end_line = f"    observation_end: {observation_end}\n" if observation_end else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
{trigger_line}{end_line}{observation_end_line}    unit: user_id
{scheme_line}
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [revenue]
"""


def _write_defs(tmp_path, trigger="saw_surface", *, allocation_scheme=None, end=None):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml(trigger, allocation_scheme=allocation_scheme, end=end))
    return p


def test_triggered_request_requires_explicit_snapshot_evidence_but_assigned_does_not(tmp_path):
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=10))
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path), con)
    try:
        con.disconnect()
        with pytest.raises(CapabilityError) as raised:
            analysis.trigger_rates()
        assert raised.value.code == "source.native.trigger_evidence_required"
    finally:
        analysis.close()


def test_snapshot_identity_distinguishes_pinned_source_evidence(tmp_path):
    path = _write_defs(tmp_path)
    cutoff = dt.datetime(2024, 1, 31, tzinfo=dt.UTC)
    later_cutoff = dt.datetime(2024, 2, 1, tzinfo=dt.UTC)
    earlier_watermark = dt.datetime(2024, 1, 30, tzinfo=dt.UTC)
    evidence = (
        (cutoff, cutoff),
        (cutoff, cutoff),
        (later_cutoff, cutoff),
        (cutoff, earlier_watermark),
    )
    analyses = []
    connections = []
    try:
        for evidence_cutoff, watermark in evidence:
            connection = ibis.duckdb.connect()
            connection.create_table("events", obj=_events(n_per_arm=100))
            connections.append(connection)
            analysis = Analysis.from_definitions(
                "exp",
                path,
                connection,
                source_snapshot_evidence=SourceSnapshotEvidence(
                    evidence_cutoff, {"events": watermark}
                ),
            )
            analyses.append(analysis)
        first, same, different_cutoff, different_watermark = [
            analysis.run(metrics=["revenue"]) for analysis in analyses
        ]
        assert all((first, same, different_cutoff, different_watermark))
        assert first[0].source_snapshot_id == same[0].source_snapshot_id
        assert first[0].source_snapshot_id != different_cutoff[0].source_snapshot_id
        assert first[0].source_snapshot_id != different_watermark[0].source_snapshot_id
        asof_first, asof_same, asof_different = [
            analysis.run_asof(metrics=["revenue"], population="assigned")
            for analysis in analyses[:3]
        ]
        assert asof_first[0].source_snapshot_id == asof_same[0].source_snapshot_id
        assert asof_first[0].source_snapshot_id != asof_different[0].source_snapshot_id
        triggered = analyses[0].run_asof(metrics=["revenue"], population="triggered")
        assert triggered[0].source_snapshot_id != asof_first[0].source_snapshot_id
    finally:
        for analysis in analyses:
            analysis.close()
        for connection in connections:
            connection.disconnect()


def test_native_pinned_identity_survives_moments_export_reload(tmp_path):
    import json

    import pyarrow.parquet as pq

    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=_events(n_per_arm=100))
    analysis = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, trigger=None),
        connection,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 31, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 30, tzinfo=dt.UTC)},
        ),
    )
    replay = changed_replay = None
    try:
        path = tmp_path / "pinned-moments.parquet"
        analysis.export(path)
        rows = pq.read_table(path).to_pylist()
        source_identity = json.loads(rows[0]["source_identity"])
        assert source_identity["source_snapshot_evidence"] == {
            "observation_cutoff_ts": "2024-01-31T00:00:00+00:00",
            "complete_through_by_feed": {
                "events": "2024-01-30T00:00:00+00:00",
            },
        }
        replay = Analysis.from_moments(
            rows,
            metrics={"revenue": "mean"},
            control="C",
        )
        replay_row = replay.run()[0]

        changed_identity = {
            **source_identity,
            "source_snapshot_evidence": {
                **source_identity["source_snapshot_evidence"],
                "observation_cutoff_ts": "2024-02-01T00:00:00+00:00",
            },
        }
        changed_rows = [
            {
                **row,
                "source_identity": json.dumps(
                    changed_identity, sort_keys=True, separators=(",", ":")
                ),
            }
            for row in rows
        ]
        changed_replay = Analysis.from_moments(
            changed_rows,
            metrics={"revenue": "mean"},
            control="C",
        )
        assert changed_replay.run()[0].source_snapshot_id != replay_row.source_snapshot_id
    finally:
        if changed_replay is not None:
            changed_replay.close()
        if replay is not None:
            replay.close()
        analysis.close()
        connection.disconnect()


def test_triggered_encouragement_uptake_remains_assignment_anchored(tmp_path):
    definitions = _defs_yaml(end='"2024-01-10"', observation_end='"2024-01-31"').replace(
        "window_days: 7", "window_days: 3"
    )
    definitions = definitions.replace(
        "      - name: saw_surface\n        column: null",
        "      - name: saw_surface\n        column: null\n      - name: clicked\n        column: null",
    )
    definitions = definitions.replace(
        "    control_group: C\n",
        "    control_group: C\n"
        "    design:\n"
        "      mechanism: encouragement\n"
        "      uptake:\n"
        "        fact: clicked\n"
        "        window_days: 7\n"
        "      one_sided: true\n"
        "      exclusion_restriction:\n"
        "        acknowledged: true\n"
        "        justification: button gates revenue\n",
    )
    path = tmp_path / "encouragement.yml"
    path.write_text(definitions)
    rows = {
        "user_id": [
            "C0",
            "T0",
            "C1",
            "T1",
            "C0",
            "T0",
            "C1",
            "T1",
            "T0",
            "T1",
            "C0",
            "T0",
            "C1",
            "T1",
        ],
        "group_id": ["C", "T", "C", "T", "C", "T", "C", "T", "T", "T", "C", "T", "C", "T"],
        "ts": [
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-02T00:00:00"),
            np.datetime64("2024-01-04T00:00:00"),
            np.datetime64("2024-01-04T00:00:00"),
            np.datetime64("2024-01-04T00:00:00"),
            np.datetime64("2024-01-04T00:00:00"),
            np.datetime64("2024-01-03T00:00:00"),
            np.datetime64("2024-01-03T00:00:00"),
            np.datetime64("2024-01-05T00:00:00"),
            np.datetime64("2024-01-05T00:00:00"),
            np.datetime64("2024-01-05T00:00:00"),
            np.datetime64("2024-01-05T00:00:00"),
        ],
        "event": [
            "enrolled",
            "enrolled",
            "enrolled",
            "enrolled",
            "saw_surface",
            "saw_surface",
            "saw_surface",
            "saw_surface",
            "clicked",
            "clicked",
            "revenue",
            "revenue",
            "revenue",
            "revenue",
        ],
        "value": [0.0] * 10 + [1.0, 2.0, 3.0, 4.0],
        "experiment_id": ["exp"] * 14,
    }
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.table(rows))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 31, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 31, tzinfo=dt.UTC)},
        ),
    )
    try:
        from tests.analysis_factory import _native_source

        moments = _native_source(analysis)._moments_for_metrics(population="triggered").to_pylist()
        by_group = {row["group_id"]: row for row in moments}
        assert by_group["T"]["sum_d"] == 2.0
        completed = analysis.run_asof(population="triggered", completed_windows_only=True)
        mature_by_day_arm = {(row.ds, row.group_id): row.n for row in completed}
        assert mature_by_day_arm[(dt.date(2024, 1, 9), "T")] == 2
        assert by_group["C"]["sum_d"] == 0.0
        triggered_asof = analysis.run_asof_lift(estimands=["compliance"], population="triggered")
        assert triggered_asof
        assert {row.estimand for row in triggered_asof} == {"compliance"}
        assert {row.analysis_population for row in triggered_asof} == {"triggered"}
        expected_history = {dt.date(2024, 1, 4) + dt.timedelta(days=offset) for offset in range(27)}
        assert {row.ds for row in triggered_asof} == expected_history
        assert all(row.require_lift().value == pytest.approx(1.0) for row in triggered_asof)
        triggered_daily = analysis.run_daily_lift(estimands=["compliance"], population="triggered")
        assert triggered_daily
        assert {row.estimand for row in triggered_daily} == {"compliance"}
        assert {row.analysis_population for row in triggered_daily} == {"triggered"}
        assert {row.ds for row in triggered_daily} == expected_history
        assert all(row.require_lift().value == pytest.approx(1.0) for row in triggered_daily)
        with pytest.raises(InvalidRequestError) as unknown_dimension:
            analysis.run_asof_lift(
                estimands=["compliance"],
                population="triggered",
                dimension="missing_dimension",
            )
        assert unknown_dimension.value.code == "facade.analysis.unknown_dimension"
        with pytest.raises(UnsupportedRequestError) as daily_dimension:
            analysis.run_daily_lift(
                estimands=["compliance"],
                population="triggered",
                dimension="missing_dimension",
            )
        assert daily_dimension.value.code == (
            "facade.analysis.daily_compliance_dimension_unsupported"
        )
    finally:
        analysis.close()


def test_empty_triggered_late_request_returns_unavailable_late_cells(tmp_path):
    definitions = (
        _defs_yaml(end='"2024-01-10"', observation_end='"2024-01-31"')
        .replace(
            "      - name: saw_surface\n        column: null",
            "      - name: saw_surface\n        column: null\n      - name: clicked\n        column: null",
        )
        .replace(
            "    control_group: C\n",
            "    control_group: C\n"
            "    design:\n"
            "      mechanism: encouragement\n"
            "      uptake: {fact: clicked, window_days: 2}\n"
            "      one_sided: true\n"
            "      exclusion_restriction: {acknowledged: true, justification: gates revenue}\n",
        )
        .replace(
            "    aggregation: sum",
            "    aggregation: sum\n    preferred_direction: increase",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    plan:\n"
            "      alternative: greater\n"
            "      secondaries:\n"
            "        - metric: revenue\n"
            "          margin: 0.25",
        )
    )
    path = tmp_path / "empty-triggered-late.yml"
    path.write_text(definitions)
    events = _events(n_per_arm=2).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 17)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(events))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        rows = analysis.run_asof_lift(population="triggered", estimands=["itt", "late"])
        assert rows
        assert {row.estimand for row in rows} == {"itt", "late"}
        assert {row.value_scale for row in rows if row.estimand == "late"} == {
            "absolute",
            "relative",
        }
        assert all(row.failure_code == "readout.cell.missing_arm" for row in rows)
        assert {row.alternative for row in rows} == {"greater"}
        assert {row.null_lift for row in rows} == {-0.25}
        assert {row.policy_name for row in rows} == {"compiled_plan"}
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_trigger_after_enrollment_end_remains_eligible_at_snapshot_cutoff(tmp_path):
    rows = {
        "user_id": ["C1", "T1", "C1", "T1", "C1", "T1"],
        "group_id": ["C", "T", "C", "T", "C", "T"],
        "ts": [
            np.datetime64("2024-01-12T09:00:00"),
            np.datetime64("2024-01-12T09:00:00"),
            np.datetime64("2024-01-17T14:00:00"),
            np.datetime64("2024-01-17T14:00:00"),
            np.datetime64("2024-01-23T09:00:00"),
            np.datetime64("2024-01-23T09:00:00"),
        ],
        "event": ["enrolled", "enrolled", "saw_surface", "saw_surface", "revenue", "revenue"],
        "value": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        "experiment_id": ["exp"] * 6,
    }
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.table(rows))
    extended_definitions = tmp_path / "extended_observation.yml"
    default_definitions = tmp_path / "default_observation.yml"
    end = "2024-01-15T00:00:00"
    extended_definitions.write_text(_defs_yaml(end=end, observation_end="2024-01-24T00:00:00"))
    default_definitions.write_text(_defs_yaml(end=end))
    evidence = SourceSnapshotEvidence(
        dt.datetime(2024, 1, 23, 12, tzinfo=dt.UTC),
        {"events": dt.datetime(2024, 1, 23, 12, tzinfo=dt.UTC)},
    )
    analysis = Analysis.from_definitions(
        "exp", extended_definitions, con, source_snapshot_evidence=evidence
    )
    default_analysis = Analysis.from_definitions(
        "exp", default_definitions, con, source_snapshot_evidence=evidence
    )
    try:
        from tests.analysis_factory import _native_source

        source = _native_source(analysis)
        grain, randomization_counts, unit_counts = source.triggered_counts()
        assert grain == "unit"
        assert randomization_counts == {"C": 1, "T": 1}
        assert unit_counts == {}
        with pytest.warns(IncrementWarning):
            extended_rows = source.unit_frame(
                analysis.metrics[0], population="triggered"
            ).to_pylist()
        assert extended_rows == []
        with pytest.warns(IncrementWarning):
            default_rows = (
                _native_source(default_analysis)
                .unit_frame(default_analysis.metrics[0], population="triggered")
                .to_pylist()
            )
        assert default_rows == []
    finally:
        analysis.close()
        default_analysis.close()


def test_unknown_trigger_refused_at_load(tmp_path):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml("nope"))
    with pytest.raises(DefinitionError) as raised:
        load(p)
    assert raised.value.code == "definition.invalid"


def test_trigger_equal_to_exposure_refused_at_load(tmp_path):
    p = tmp_path / "defs.yml"
    p.write_text(_defs_yaml("assignment"))
    with pytest.raises(DefinitionError) as raised:
        load(p)
    assert raised.value.code == "definition.invalid"


def test_trigger_declaration_loads(tmp_path):
    defs = load(_write_defs(tmp_path))
    exp = defs.experiment("exp")
    assert exp is not None
    assert exp.trigger == "saw_surface"


def _events(
    n_per_arm=2000,
    trigger_rate=0.2,
    effect=0.5,
    seed=5,
    treatment_trigger_rate=None,
):
    """Only triggered treatment units are affected. The assigned-population
    lift is therefore the true lift times the trigger rate."""

    rng = np.random.default_rng(seed)
    rows = {"user_id": [], "group_id": [], "ts": [], "event": [], "value": [], "experiment_id": []}
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    revenue_ts = np.datetime64("2024-01-08T00:00:00")
    freshness_ts = np.datetime64("2024-01-09T00:00:00")

    def add(uid, arm, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        arm_trigger_rate = (
            treatment_trigger_rate
            if arm == "T" and treatment_trigger_rate is not None
            else trigger_rate
        )
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            triggered = rng.random() < arm_trigger_rate
            add(uid, arm, "enrolled", 0.0, enrolled_ts)
            if triggered:
                add(uid, arm, "saw_surface", 0.0, enrolled_ts)
            base = rng.lognormal(0.0, 0.4)
            lift = effect if (arm == "T" and triggered) else 0.0
            add(uid, arm, "revenue", base * (1.0 + lift), revenue_ts)
            add(uid, arm, "revenue", base * (1.0 + lift), freshness_ts)
    return pa.table(rows)


def _with_preassignment_region(table: pa.Table) -> pa.Table:
    user_ids = sorted(set(table["user_id"].to_pylist()))
    arms = {user: user[0] for user in user_ids}
    regions = {user: ("north" if int(user[1:]) % 2 else "south") for user in user_ids}
    table = table.append_column(
        "region", pa.array([regions[user] for user in table["user_id"].to_pylist()])
    )
    profiles = pa.table(
        {
            "user_id": user_ids,
            "group_id": [arms[user] for user in user_ids],
            "ts": pa.array(
                [dt.datetime(2024, 1, 1)] * len(user_ids), type=table.schema.field("ts").type
            ),
            "event": ["profile"] * len(user_ids),
            "value": [0.0] * len(user_ids),
            "experiment_id": ["exp"] * len(user_ids),
            "region": [regions[user] for user in user_ids],
        }
    )
    return pa.concat_tables([table, profiles])


def _assert_parity_tables_close(left: pa.Table, right: pa.Table, *, keys: tuple[str, ...]) -> None:
    left_rows = {tuple(row[key] for key in keys): row for row in left.to_pylist()}
    right_rows = {tuple(row[key] for key in keys): row for row in right.to_pylist()}
    assert left_rows.keys() == right_rows.keys()
    for key, left_row in left_rows.items():
        right_row = right_rows[key]
        assert left_row.keys() == right_row.keys()
        for name, value in left_row.items():
            other = right_row[name]
            if isinstance(value, float):
                assert value == pytest.approx(other, rel=1e-12, abs=1e-12)
            else:
                assert value == other


def _analysis(
    tmp_path, table, trigger="saw_surface", *, on_mixed_assignment="error", allocation_scheme=None
):
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    return Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, trigger, allocation_scheme=allocation_scheme),
        con,
        on_mixed_assignment=on_mixed_assignment,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )


def test_reports_both_populations_and_the_triggered_effect_is_undiluted(tmp_path):
    rate, effect = 0.2, 0.5
    an = _analysis(tmp_path, _events(trigger_rate=rate, effect=effect))
    rows = an.run()

    by_pop = {r.analysis_population: r for r in rows if r.metric == "revenue"}
    assert set(by_pop) == {"assigned", "triggered"}

    assigned, triggered = by_pop["assigned"], by_pop["triggered"]
    # The assigned reading is the true effect diluted by the trigger rate.
    assert assigned.require_lift().value == pytest.approx(effect * rate, rel=0.25)
    # The triggered reading recovers the real thing.
    assert triggered.require_lift().value == pytest.approx(effect, rel=0.12)
    assert triggered.require_lift().value > assigned.require_lift().value
    assert {row.multiplicity_status for row in rows} == {"declared_plan"}
    families = rows.metadata.scope.families
    secondary = [family for family in families if family.name == "secondary"]
    assert {family.analysis_population for family in secondary} == {"assigned", "triggered"}
    assert len({family.family_id for family in secondary}) == 2
    triggered_rows = rows.filter(lambda row: row.analysis_population == "triggered")
    assert {row.multiplicity_status for row in triggered_rows} == {"declared_plan"}
    assert triggered_rows.metadata.scope.families == families

    # The table adapter must carry the population axis through the same
    # assigned/triggered result pair; otherwise the two rows collide in the
    # rendered metric key.
    from increment.tables import estimates_to_readout, readout_table

    readout_rows = estimates_to_readout([assigned, triggered])
    assert [row["analysis_population"] for row in readout_rows] == ["assigned", "triggered"]
    pytest.importorskip("coeftable")
    html = readout_table(readout_rows).gt().as_raw_html()
    assert "(assigned)" in html
    assert "(triggered)" in html


def test_triggered_multiplicity_keeps_failed_cells_in_fixed_roster_across_ingresses(
    tmp_path,
):
    table_rows = _events(n_per_arm=2000, trigger_rate=0.2, effect=0.5).to_pylist()
    table_rows.extend(
        {
            **row,
            "event": "partial",
        }
        for row in tuple(table_rows)
        if row["group_id"] == "T" and row["event"] == "revenue"
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(table_rows))
    definitions = (
        _defs_yaml()
        .replace(
            "      - name: enrolled\n",
            "      - name: partial\n        column: value\n"
            "      - name: missing\n        column: value\n"
            "      - name: enrolled\n",
        )
        .replace(
            "experiments:\n",
            "  - name: partial\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: partial\n"
            "    aggregation: sum\n"
            "    window_days: 7\n"
            "  - name: missing\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: missing\n"
            "    aggregation: sum\n"
            "    window_days: 7\n"
            "experiments:\n",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    plan:\n"
            "      view_multiplicity: {correction: bonferroni}\n"
            "      secondaries: [revenue, partial, missing]",
        )
    )
    path = tmp_path / "triggered_multiplicity.yml"
    path.write_text(definitions)
    evidence = SourceSnapshotEvidence(
        dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
        {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
    )
    analysis = Analysis.from_definitions("exp", path, con, source_snapshot_evidence=evidence)
    reopened = None
    try:
        from increment.query.artifact_contract import unit_day_artifact_extension_catalog
        from increment.query.artifact_publish import artifact_context
        from increment.semantics.loader import load

        context = artifact_context(load(path), analysis.experiment, "error")
        extensions = [
            entry.request
            for entry in unit_day_artifact_extension_catalog(context)
            if entry.request.kind
            in {"trigger_population", "assignment_counts", "trigger_measure_stats"}
        ]
        from increment.query.session import WarehouseArtifactStore

        store = WarehouseArtifactStore(con, schema_name="triggered_multiplicity")
        reference = analysis.publish_unit_day_artifact(store, extensions=extensions)
        reopened = Analysis.from_unit_day_artifact(store, reference, expected_context=context)

        native_rows = analysis.run(metrics=["revenue", "partial", "missing"])
        artifact_rows = reopened.run(metrics=["revenue", "partial", "missing"])
        for rows in (native_rows, artifact_rows):
            triggered = rows.filter(lambda row: row.analysis_population == "triggered")
            family = next(
                scope
                for scope in triggered.metadata.scope.families
                if scope.name == "secondary" and scope.analysis_population == "triggered"
            )
            assert all(cell.analysis_population == "triggered" for cell in family.members)
            assert len(family.members) == 3
            assert {cell.metric for cell in family.members} == {
                "revenue",
                "partial",
                "missing",
            }
            assert {cell.group_id for cell in family.members} == {"T"}
            assert family.complete is True
            missing = next(row for row in triggered if row.metric == "missing")
            assert missing.relative_unavailable_reason == "nonpositive_arm_mean"
            revenue_treatment = next(
                row for row in triggered if row.metric == "revenue" and row.group_id == "T"
            )
            assert revenue_treatment.family_size == 3
            assert revenue_treatment.family_threshold == pytest.approx(0.1 / 3)
            assert revenue_treatment.require_lift().value == pytest.approx(0.5, rel=0.12)

        native_triggered = {
            (row.metric, row.group_id): row
            for row in native_rows
            if row.analysis_population == "triggered"
        }
        artifact_triggered = {
            (row.metric, row.group_id): row
            for row in artifact_rows
            if row.analysis_population == "triggered"
        }
        assert set(native_triggered) == set(artifact_triggered)
        assert {key: row.family_threshold for key, row in native_triggered.items()} == {
            key: row.family_threshold for key, row in artifact_triggered.items()
        }
        assert {key: row.failure_code for key, row in native_triggered.items()} == {
            key: row.failure_code for key, row in artifact_triggered.items()
        }
    finally:
        analysis.close()
        if reopened is not None:
            reopened.close()
        con.disconnect()


def test_observed_trigger_rate_is_reported(tmp_path):
    """The planner needs the measured rate to size a follow-up."""
    an = _analysis(tmp_path, _events(trigger_rate=0.2))
    rates = an.trigger_rates()
    assert set(rates) == {"C", "T"}
    for arm in ("C", "T"):
        assert rates[arm] == pytest.approx(0.2, abs=0.03)


@pytest.mark.parametrize(
    ("policy", "warning"),
    [("error", False), ("warn", True), ("exclude", False)],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_trigger_rates_enforces_assignment_policy(tmp_path, policy, warning):
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(trigger_rate=0.2))
    an = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path),
        con,
        on_mixed_assignment=policy,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    con.raw_sql("UPDATE events SET group_id = NULL WHERE user_id = 'C0' AND event = 'enrolled'")

    if policy == "error":
        with pytest.raises(InvalidRequestError) as caught:
            an.trigger_rates()
        assert caught.value.code == "query.integrity.unassigned_assignment_units"
        return

    if warning:
        with pytest.warns(IncrementWarning) as rec:
            rates = an.trigger_rates()
        assert "query.integrity.mixed_assignments_excluded" in warning_codes(rec)
    else:
        rates = an.trigger_rates()
    assert set(rates) == {"C", "T"}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triggered_assignment_counts_exclude_assigned_audit_counts(tmp_path):

    rows = _events(trigger_rate=0.2).to_pylist()
    assignments = [row for row in rows if row["event"] == "enrolled"]
    mixed_row = dict(assignments[0])
    mixed_row["group_id"] = "T" if mixed_row["group_id"] == "C" else "C"
    unassigned_row = dict(assignments[1])
    unassigned_row["group_id"] = None
    rows.extend((mixed_row, unassigned_row))

    analysis = _analysis(tmp_path, pa.Table.from_pylist(rows), on_mixed_assignment="exclude")
    try:
        assert analysis.trigger_rates() == pytest.approx({"C": 0.20270270270270271, "T": 0.1925})
    finally:
        analysis.close()


def test_one_arm_trigger_refused(tmp_path):
    """A trigger only the treatment arm can produce is the treatment."""
    table = _events(trigger_rate=0.2)
    events = table.column("event").to_pylist()
    groups = table.column("group_id").to_pylist()
    keep = [
        i for i in range(table.num_rows) if not (events[i] == "saw_surface" and groups[i] == "C")
    ]
    an = _analysis(tmp_path, table.take(keep))
    with pytest.raises(InvalidRequestError) as caught:
        an.run()
    assert caught.value.code == "query.integrity.trigger_arm_missing"


def test_no_trigger_declared_leaves_every_row_assigned(tmp_path):
    """Backwards compatibility: an experiment with no trigger behaves
    exactly as before and reports one row per metric."""
    an = _analysis(tmp_path, _events(), trigger=None)
    rows = [r for r in an.run() if r.metric == "revenue"]
    assert len(rows) == 1
    assert rows[0].analysis_population == "assigned"


def _assert_population_scoped_consumers(rows):
    from increment import segment_heterogeneity, segment_rollout_recommendation
    from increment.breakout.estimates import BreakoutEstimates

    by_population = {
        population: [row for row in rows if row.analysis_population == population]
        for population in ("assigned", "triggered")
    }
    assert all(len(population_rows) == 2 for population_rows in by_population.values())
    heterogeneity = segment_heterogeneity(rows)
    by_population_heterogeneity = [
        segment_heterogeneity(BreakoutEstimates(population_rows))
        for population_rows in by_population.values()
    ]
    assert len(heterogeneity.summary) == sum(
        len(result.summary) for result in by_population_heterogeneity
    )
    assert {row.analysis_population for row in heterogeneity.summary} == {"assigned", "triggered"}
    assert {row.analysis_population for row in heterogeneity.segments} == {"assigned", "triggered"}
    assert "analysis_population" in heterogeneity.summary.to_frame(backend="polars").columns
    assert "analysis_population" in heterogeneity.segments.to_frame(backend="polars").columns

    rollout, rollout_segments = segment_rollout_recommendation(rows)
    by_population_rollout = [
        segment_rollout_recommendation(BreakoutEstimates(population_rows))
        for population_rows in by_population.values()
    ]
    assert len(rollout) == sum(len(result[0]) for result in by_population_rollout)
    assert len(rollout_segments) == sum(len(result[1]) for result in by_population_rollout)
    assert {row.analysis_population for row in rollout} == {"assigned", "triggered"}
    assert {row.analysis_population for row in rollout_segments} == {"assigned", "triggered"}
    return by_population


def test_run_breakout_returns_distinct_triggered_population_and_families(tmp_path):
    table = _with_preassignment_region(_events(n_per_arm=300, trigger_rate=0.5, effect=0.6))
    path = tmp_path / "defs.yml"
    path.write_text(_defs_yaml())
    path.write_text(
        path.read_text()
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n"
            "    factors: [{property: region, source: events}]\n"
            "    plan:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        native_summaries = analysis.breakout_summaries()
        native_factors = analysis.factor_summaries()
        assert {key.rsplit(":", 1)[1] for key in native_summaries} == {"assigned", "triggered"}
        for key, tables in native_summaries.items():
            population = key.rsplit(":", 1)[1]
            assert all("analysis_population" in table.column_names for table in tables.values())
            assert all(
                set(table["analysis_population"].to_pylist()) == {population}
                for table in tables.values()
            )
        rows = analysis.run_breakout()
        assert "analysis_population" in rows.to_frame(backend="polars").columns
        assert {row.analysis_population for row in rows} == {"assigned", "triggered"}
        assert {row.dimension_value for row in rows} == {"north", "south"}
        assert {row.dimension for row in rows} == {"region"}
        by_population = _assert_population_scoped_consumers(rows)
        assert all(
            row.n_control is not None and 0 < row.n_control < 150
            for row in by_population["triggered"]
        )
        assert all(
            row.n_treat is not None and 0 < row.n_treat < 150 for row in by_population["triggered"]
        )
        assert (
            len(
                {
                    row.family_id
                    for population_rows in by_population.values()
                    for row in population_rows
                }
            )
            == 2
        )
        assert rows.metadata is not None
        assert {family.analysis_population for family in rows.metadata.scope.families} == {
            "assigned",
            "triggered",
        }
        from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=(
                "breakout_dimension",
                "factor_dimension",
                "assignment_counts",
                "trigger_measure_stats",
            ),
        )
        artifact_factors = adopted.factor_summaries()
        assert native_factors.keys() == artifact_factors.keys()
        for key, table in native_factors.items():
            _assert_parity_tables_close(
                table,
                artifact_factors[key],
                keys=("analysis_population", "group_id", "region"),
            )
        artifact_summaries = adopted.breakout_summaries()
        assert native_summaries.keys() == artifact_summaries.keys()
        for key, tables in native_summaries.items():
            assert tables.keys() == artifact_summaries[key].keys()
            for name, table in tables.items():
                _assert_parity_tables_close(
                    table,
                    artifact_summaries[key][name],
                    keys=tuple(
                        column
                        for column in ("analysis_population", "group_id", "region", "ds", "cohort")
                        if column in table.column_names
                    ),
                )
        artifact_rows = adopted.run_breakout()
        native_by_cell = {
            (row.analysis_population, row.dimension_value, row.group_id): row for row in rows
        }
        artifact_by_cell = {
            (row.analysis_population, row.dimension_value, row.group_id): row
            for row in artifact_rows
        }
        assert native_by_cell.keys() == artifact_by_cell.keys()
        for cell, native_row in native_by_cell.items():
            artifact_row = artifact_by_cell[cell]
            assert artifact_row.n_control == native_row.n_control
            assert artifact_row.n_treat == native_row.n_treat
            assert artifact_row.require_lift().value == pytest.approx(
                native_row.require_lift().value
            )
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_artifact_windowed_quantile_refuses_before_evidence_read(tmp_path):
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=_with_preassignment_region(_events(n_per_arm=30, trigger_rate=0.5, effect=0.6)),
    )
    path = tmp_path / "windowed-quantile.yml"
    path.write_text(
        _defs_yaml()
        .replace("    type: mean\n", "    type: quantile\n    quantile: 0.5\n")
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
        )
    )
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        adopted = _publish_and_adopt(
            con, analysis, kinds=("breakout_dimension", "assignment_counts")
        )
        with pytest.raises(InvalidRequestError) as raised:
            adopted.run_breakout(population="triggered")
        assert raised.value.code == "frame.metric.window_days_supported"
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_artifact_without_breakouts_returns_empty_estimates(tmp_path):
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

    path = _write_defs(tmp_path)
    path.write_text(
        path.read_text().replace("    type: mean\n", "    type: quantile\n    quantile: 0.5\n")
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=10))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=("trigger_population", "assignment_counts", "trigger_measure_stats"),
        )
        assert adopted.run_breakout() == []
        assert adopted.run_breakout(population="assigned") == []
        assert adopted.run_breakout(population="triggered") == []
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_artifact_run_refuses_invalid_estimand_before_population_read(
    tmp_path, monkeypatch
):
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=10))
    analysis = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    armed = False
    try:
        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=("trigger_population", "assignment_counts", "trigger_measure_stats"),
        )
        query_calls = []
        for method in ("execute", "to_pyarrow"):
            original = getattr(con, method, None)
            if original is not None:

                def guard_query(*args, _method=method, _original=original, **kwargs):
                    if armed:
                        query_calls.append(_method)
                        raise AssertionError("invalid request must not query the artifact")
                    return _original(*args, **kwargs)

                monkeypatch.setattr(con, method, guard_query)
        armed = True
        with pytest.raises(InvalidRequestError) as refused:
            adopted.run(population="triggered", estimands=["late"])
        assert refused.value.code == "readout.assignment.estimands"
        assert query_calls == []
    finally:
        armed = False
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_artifact_breakout_refuses_invalid_method_before_population_read(
    tmp_path, monkeypatch
):
    from increment.estimation.engine import Method
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

    definitions = (
        _defs_yaml()
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_with_preassignment_region(_events(n_per_arm=10)))
    path = tmp_path / "triggered-breakout.yml"
    path.write_text(definitions)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    armed = False
    try:
        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=(
                "breakout_dimension",
                "trigger_population",
                "assignment_counts",
                "trigger_measure_stats",
            ),
        )
        query_calls = []
        for method in ("execute", "to_pyarrow"):
            original = getattr(con, method, None)
            if original is not None:

                def guard_query(*args, _method=method, _original=original, **kwargs):
                    if armed:
                        query_calls.append(_method)
                        raise AssertionError("invalid request must not query the artifact")
                    return _original(*args, **kwargs)

                monkeypatch.setattr(con, method, guard_query)
        armed = True
        with pytest.raises(InvalidRequestError) as refused:
            adopted.run_breakout(
                population="triggered",
                decision_method=Method(name="iptw"),
            )
        assert refused.value.code == "estimation.engine.method_name_observational"
        assert query_calls == []
    finally:
        armed = False
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triggered_breakout_summary_preserves_zero_event_days(tmp_path):
    events = _events(n_per_arm=2, trigger_rate=1.0).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 3)
        elif row["event"] == "revenue":
            row["ts"] = dt.datetime(2024, 1, 5)
    table = _with_preassignment_region(pa.Table.from_pylist(events))
    path = tmp_path / "zero-days.yml"
    path.write_text(
        _defs_yaml(end='"2024-01-04T00:00:00"', observation_end='"2024-01-31T00:00:00"')
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        native = analysis.breakout_summaries()
        native_daily = native["revenue:region:events:triggered"]["daily_group_summary"]
        zero_days = {dt.date(2024, 1, day) for day in (6, 7, 8, 9)}
        native_rows = native_daily.to_pylist()
        assert zero_days <= {row["ds"] for row in native_rows}
        assert all(
            row["n"] > 0 and row["ref_y"] == row["cy1"] == row["cy2"] == 0
            for row in native_rows
            if row["ds"] in zero_days
        )
        from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=("breakout_dimension", "trigger_population", "trigger_measure_stats"),
        )
        artifact_daily = adopted.breakout_summaries()["revenue:region:events:triggered"][
            "daily_group_summary"
        ]
        _assert_parity_tables_close(
            native_daily,
            artifact_daily,
            keys=("analysis_population", "group_id", "region", "ds"),
        )
        assert zero_days <= {row["ds"] for row in artifact_daily.to_pylist()}
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triggered_breakout_summary_stops_at_pinned_cutoff(tmp_path):
    events = _events(n_per_arm=2, trigger_rate=1.0).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 3)
    for arm in ("C", "T"):
        for unit in range(2):
            events.append(
                {
                    "user_id": f"{arm}{unit}",
                    "group_id": arm,
                    "ts": dt.datetime(2024, 1, 4),
                    "event": "revenue",
                    "value": 1.0,
                    "experiment_id": "exp",
                }
            )
    table = _with_preassignment_region(pa.Table.from_pylist(events))
    path = tmp_path / "cutoff.yml"
    path.write_text(
        _defs_yaml(end='"2024-01-04T00:00:00"', observation_end='"2024-01-31T00:00:00"')
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 5, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 5, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        with pytest.warns(IncrementWarning):
            native = analysis.breakout_summaries()
        native_daily = native["revenue:region:events:triggered"]["daily_group_summary"]
        native_days = native_daily["ds"].to_pylist()
        assert native_days
        assert max(native_days) <= dt.date(2024, 1, 4)
        from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=("breakout_dimension", "trigger_population", "trigger_measure_stats"),
        )
        artifact = adopted.breakout_summaries()
        artifact_daily = artifact["revenue:region:events:triggered"]["daily_group_summary"]
        assert max(artifact_daily["ds"].to_pylist()) <= dt.date(2024, 1, 4)
        _assert_parity_tables_close(
            native_daily,
            artifact_daily,
            keys=("analysis_population", "group_id", "region", "ds"),
        )
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_cate_uses_native_unit_population(tmp_path):
    events = _events(n_per_arm=300, trigger_rate=0.5, effect=0.6).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 3)
    table = _with_preassignment_region(pa.Table.from_pylist(events))
    path = _write_defs(tmp_path)
    path.write_text(
        path.read_text().replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        from increment.estimation.cate import Covariate
        from increment.query.artifact_contract import ArtifactContractError
        from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

        interact = [Covariate(name="region", kind="categorical")]
        estimate = analysis.estimate_cate("revenue", control="C", interact=interact)
        validation = analysis.validate_cate("revenue", control="C", interact=interact, n_groups=3)
        targeting = analysis.targeting_rule("revenue", control="C", interact=interact, fraction=0.5)
        selection = analysis.select_targeting_rule(
            "revenue",
            control="C",
            interact=interact,
            fractions=[0.5],
            seed=7,
            n_folds=3,
        )
        assert estimate.n < 600
        assert estimate.n_treated + estimate.n_control == estimate.n
        assert validation is not None
        assert targeting is not None
        assert selection is not None
        adopted = _publish_and_adopt(
            con, analysis, kinds=("trigger_population", "trigger_measure_stats")
        )
        with pytest.raises(ArtifactContractError) as raised:
            adopted.estimate_cate("revenue", control="C", interact=interact)
        assert raised.value.code == "artifact.extension.missing"
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_ratio_breakout_daily_summary_uses_trigger_anchors(tmp_path):
    events = _events(n_per_arm=2, trigger_rate=1.0).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 5)
    for arm in ("C", "T"):
        for unit in range(2):
            user = f"{arm}{unit}"
            for day in (9, 10, 11):
                for fact, value in (("revenue", 2.0), ("session", 1.0)):
                    events.append(
                        {
                            "user_id": user,
                            "group_id": arm,
                            "ts": dt.datetime(2024, 1, day),
                            "event": fact,
                            "value": value,
                            "experiment_id": "exp",
                        }
                    )
    table = _with_preassignment_region(pa.Table.from_pylist(events))
    path = _write_defs(tmp_path)
    path.write_text(
        path.read_text()
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "      - name: saw_surface\n        column: null\n",
            "      - name: saw_surface\n        column: null\n"
            "      - name: session\n        column: null\n",
        )
        .replace(
            "experiments:\n",
            "  - name: revenue_per_session\n"
            "    type: ratio\n"
            "    entity: user_id\n"
            "    numerator: {fact: revenue, aggregation: sum, window_days: 7}\n"
            "    denominator: {fact: session, aggregation: count, window_days: 7}\n"
            "experiments:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n    plan:\n",
        )
        .replace("secondaries: [revenue]", "secondaries: [revenue, revenue_per_session]")
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 20, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 20, tzinfo=dt.UTC)},
        ),
    )
    adopted = None
    try:
        native = analysis.breakout_summaries(metrics=["revenue_per_session"])
        native_daily = native["revenue_per_session:region:events:triggered"]["daily_group_summary"]
        assert {dt.date(2024, 1, day) for day in (9, 10, 11)} <= set(native_daily["ds"].to_pylist())
        from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt

        adopted = _publish_and_adopt(
            con,
            analysis,
            kinds=("breakout_dimension", "trigger_population", "trigger_measure_stats"),
        )
        artifact = adopted.breakout_summaries(metrics=["revenue_per_session"])
        artifact_daily = artifact["revenue_per_session:region:events:triggered"][
            "daily_group_summary"
        ]
        _assert_parity_tables_close(
            native_daily,
            artifact_daily,
            keys=("analysis_population", "group_id", "region", "ds"),
        )
    finally:
        if adopted is None:
            analysis.close()
            con.disconnect()
        else:
            _close_parity_analysis(adopted)


def test_triggered_factor_group_and_allocation_preserve_population(tmp_path):
    events = _events(n_per_arm=300, trigger_rate=0.5, effect=0.6).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] += dt.timedelta(days=1)
    table = _with_preassignment_region(pa.Table.from_pylist(events))
    path = _write_defs(tmp_path)
    path.write_text(
        path.read_text()
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    factors: [{property: region}]\n    plan:\n",
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=table)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        factors = analysis.factor_summaries()
        assert {key.rsplit(":", 1)[1] for key in factors} == {"assigned", "triggered"}
        assert all("analysis_population" in table.column_names for table in factors.values())
        assert all(
            set(table["region"].to_pylist()) == {"north", "south"} for table in factors.values()
        )
        assert {
            value
            for table in factors.values()
            for value in table["analysis_population"].to_pylist()
        } == {"assigned", "triggered"}

        assigned = analysis.dashboard_group_data(metrics=analysis.metrics)
        triggered = analysis.dashboard_group_data(metrics=analysis.metrics, population="triggered")
        assert assigned and triggered
        assert {row.analysis_population for row in assigned} == {"assigned"}
        assert {row.analysis_population for row in triggered} == {"triggered"}
        assert all(row.assigned_units is not None and row.assigned_units < 300 for row in triggered)

        assigned_history = analysis.allocation_history()
        triggered_history = analysis.allocation_history(population="triggered")
        assert "analysis_population" in triggered_history.column_names
        assert set(triggered_history["analysis_population"].to_pylist()) == {"triggered"}
        assert sum(triggered_history["n_daily"].to_pylist()) < sum(
            assigned_history["n_daily"].to_pylist()
        )
        assert set(assigned_history["ds"].to_pylist()) == {dt.date(2024, 1, 2)}
        assert set(triggered_history["ds"].to_pylist()) == {dt.date(2024, 1, 3)}
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_sitewide_refuses_with_source_specific_reason(tmp_path):
    analysis = _analysis(tmp_path, _events(n_per_arm=100))
    with pytest.raises(CapabilityError) as raised:
        analysis.sitewide("revenue")
    assert raised.value.code == "analysis.sitewide.triggered_population_unsupported"


def test_triggered_sitewide_refusal_from_seam_keeps_coded_source_reason():
    import pandas as pd

    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "user_id": ["c1", "c2", "t1", "t2"],
                "variant": ["control", "control", "treatment", "treatment"],
                "revenue": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        analysis.sitewide("revenue")
    assert raised.value.code == "facade.analysis.operation"


@pytest.mark.parametrize("population", ["assigned", "triggered"])
@pytest.mark.parametrize(
    "method_name",
    ("run_daily", "run_daily_lift", "run_asof", "run_asof_lift"),
)
def test_triggered_experiment_day_axis_readouts_retain_population_identity(
    tmp_path, method_name, population
):
    analysis = _analysis(tmp_path, _events(n_per_arm=100))
    try:
        rows = getattr(analysis, method_name)(population=population)
        assert rows
        assert {row.analysis_population for row in rows} == {population}
        assert rows.metadata is not None
        assert rows.metadata.scope.populations == (population,)
        assert {cell.analysis_population for cell in rows.metadata.scope.cells} == {population}
        frame_rows = rows.to_frame(backend="pyarrow").to_pylist()
        assert {row["analysis_population"] for row in frame_rows} == {population}
        from increment.estimation.readout_types import ReadoutResults

        restored = ReadoutResults.model_validate_json(rows.model_dump_json())
        assert {row.analysis_population for row in restored} == {population}
        assert restored.metadata == rows.metadata
        if method_name in ("run_daily_lift", "run_asof_lift"):
            from increment.tables import estimates_to_readout

            assert {row["analysis_population"] for row in estimates_to_readout(rows)} == {
                population
            }
    finally:
        analysis.close()


def test_srm_population_triggered_counts_the_narrowed_population(tmp_path):
    """srm(population='triggered') is a real guardrail: a trigger
    correlated with assignment shows up as an allocation imbalance there
    even when enrollment itself is balanced."""
    an = _analysis(
        tmp_path,
        _events(trigger_rate=0.2, n_per_arm=500),
        allocation_scheme="independent",
    )
    expected = {"C": 0.5, "T": 0.5}
    assigned = an.srm(expected=expected)
    triggered = an.srm(expected=expected, population="triggered")
    assert sum(assigned.observed.values()) == 1000
    assert sum(triggered.observed.values()) < sum(assigned.observed.values())


def test_triggered_run_does_not_retest_assigned_law_on_selected_counts(tmp_path):
    analysis = _analysis(
        tmp_path,
        _events(n_per_arm=500, trigger_rate=0.1, treatment_trigger_rate=0.9),
        allocation_scheme="independent",
    )

    results = analysis.run()

    scopes = {
        scope.rosters[0].analysis_population: scope
        for scope in results.metadata.scope.by_source.values()
    }
    assert scopes["assigned"].integrity[0].status == "not_rejected"
    assert scopes["assigned"].integrity[0].analysis_population == "assigned"
    (triggered_integrity,) = scopes["triggered"].integrity
    assert triggered_integrity.status == "not_checked_missing_counts"
    assert triggered_integrity.analysis_population == "triggered"
    assert triggered_integrity.observed is None
    triggered_only = results.filter(lambda row: row.analysis_population == "triggered")
    retained = next(
        scope
        for scope in triggered_only.metadata.scope.by_source.values()
        if scope.rosters[0].analysis_population == "assigned"
    )
    assert retained.integrity[0] == scopes["assigned"].integrity[0]


def test_registered_sequential_trigger_run_reuses_assigned_integrity(tmp_path, monkeypatch):
    import datetime as dt

    from increment.readouts import _sequential_scope
    from tests.binary_sequential_cases import definitions_yaml, event_rows, unit_rows
    from tests.sequential_cases import registered_native

    units = unit_rows(seed=17, n=500)
    rng = np.random.default_rng(51)
    for unit in units:
        threshold = 0.1 if unit["variant"] == "control" else 0.9
        unit["triggered"] = rng.random() < threshold
    events = event_rows(units)
    trigger_time = dt.datetime(2025, 1, 10, 10, tzinfo=dt.UTC)
    events.extend(
        {
            "user_id": unit["user_id"],
            "ts": trigger_time,
            "event": "saw_surface",
            "experiment_id": None,
            "group_id": None,
        }
        for unit in units
        if unit["triggered"]
    )
    definitions = (
        definitions_yaml("duckdb", "events", plan="      primary: purchase\n")
        .replace(
            "      - {name: buy, column: null}",
            "      - {name: buy, column: null}\n      - {name: saw_surface, column: null}",
        )
        .replace(
            "  - {name: enrollment, fact: enrolled}",
            "  - {name: enrollment, fact: enrolled}\n  - {name: saw_surface, fact: saw_surface}",
        )
        .replace(
            "    allocation: {control: 0.5, treatment: 0.5}",
            "    allocation: {control: 0.5, treatment: 0.5}\n"
            "    allocation_scheme: independent\n    trigger: saw_surface",
        )
    )
    definitions_path = tmp_path / "sequential-trigger.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(events))
    query_calls = []
    armed = False
    for method in ("execute", "to_pyarrow"):
        original = getattr(con, method, None)
        if original is not None:

            def guard_query(*args, _method=method, _original=original, **kwargs):
                if armed:
                    query_calls.append(_method)
                    raise AssertionError("run() must not query the warehouse after capture")
                return _original(*args, **kwargs)

            monkeypatch.setattr(con, method, guard_query)
    analysis = registered_native(Analysis.from_definitions("exp", definitions_path, con))
    analysis.capture_sequential(finalized=True, as_of=dt.date(2025, 1, 14))
    snapshot = analysis.sequential_snapshot()
    triggered_history = analysis.run_asof_lift(population="triggered")
    assert triggered_history
    assert not any(
        family.analysis_population == "triggered"
        for family in triggered_history.metadata.scope.families
    )
    assert {row.analysis_population for row in triggered_history} == {"triggered"}
    assert {row.failure_code for row in triggered_history} == {"readout.cell.unsupported_request"}
    assert {row.failure_context["reason"] for row in triggered_history} == {"triggered_sequential"}
    selection = [
        (
            row.discovery,
            row.family_id,
            row.family_axes,
            row.family_q,
            row.family_threshold,
            row.family_guarantee,
            row.family_nominal_alpha,
        )
        for row in triggered_history
    ]
    assert selection == [(None,) * 7] * len(triggered_history)
    assert all(row.sequential_result is None for row in triggered_history)
    assert {
        cell.failure.code for cell in triggered_history.metadata.cells if cell.failure is not None
    } == {"readout.cell.unsupported_request"}

    assert snapshot.assignment_counts == {"control": 500, "treatment": 500}
    integrity_calls = []
    original_integrity = _sequential_scope.assignment_integrity

    def counted_integrity(*args, **kwargs):
        integrity_calls.append(kwargs.get("counts", args[1] if len(args) > 1 else None))
        return original_integrity(*args, **kwargs)

    monkeypatch.setattr(_sequential_scope, "assignment_integrity", counted_integrity)
    armed = True
    results = analysis.run()

    (scope,) = results.metadata.scope.by_source.values()
    (integrity,) = scope.integrity
    assert query_calls == []
    assert len(integrity_calls) == 1
    assert integrity.analysis_population == "assigned"
    assert integrity.observed == snapshot.assignment_counts
    assert integrity.observed == {"control": 500, "treatment": 500}
    assert any(
        row.analysis_population == "triggered"
        and row.failure_code == "readout.cell.unsupported_request"
        for row in results
    )
    assert any(
        component["kind"] == "assignment_counts" for component in results.source["components"]
    )


def test_srm_population_triggered_refuses_on_a_seam_instance():
    import pandas as pd

    from increment import Analysis
    from increment.errors import CapabilityError

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    a = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    with pytest.raises(CapabilityError) as raised:
        a.srm(population="triggered")
    assert raised.value.code == "facade.analysis.operation"


def test_srm_rejects_unknown_population_before_assigned_readout():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "variant": ["treatment", "treatment", "control", "control"],
            "revenue": [25.0, 35.0, 22.5, 13.0],
        }
    )
    analysis = Analysis.from_unit_summary(
        df, unit="user_id", group="variant", control="control", metrics={"revenue": "mean"}
    )
    with pytest.raises(InvalidRequestError) as caught:
        analysis.srm(population="trigger")  # ty: ignore[invalid-argument-type]
    assert caught.value.code == "facade.analysis.invalid_population"
    assert caught.value.context["population"] == "trigger"


def _defs_yaml_clustered(trigger: str | None = "saw_surface") -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM cluster_trigger_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: revenue
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: revenue
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: assignment
{trigger_line}    unit: user_id
    cluster: store_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [revenue]
"""


def _clustered_trigger_events(
    n_stores_per_arm=20,
    units_per_store=5,
    trigger_rate=0.2,
    effect=0.5,
    seed=5,
):
    """Same dilution shape as `_events`, but store-clustered: every row
    also carries a `store_id`, and a store's units share triggered status
    so the population narrowing and the cluster grouping are both live."""

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "user_id": [],
        "group_id": [],
        "store_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    revenue_ts = np.datetime64("2024-01-08T00:00:00")
    freshness_ts = np.datetime64("2024-01-09T00:00:00")

    def add(uid, arm, store, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["store_id"].append(store)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        for s in range(n_stores_per_arm):
            store = f"{arm}{s}"
            store_triggered = rng.random() < trigger_rate
            arm_units_per_store = (
                units_per_store[arm] if isinstance(units_per_store, dict) else units_per_store
            )
            for u in range(arm_units_per_store):
                uid = f"{store}_{u}"
                add(uid, arm, store, "enrolled", 0.0, enrolled_ts)
                if store_triggered:
                    add(uid, arm, store, "saw_surface", 0.0, enrolled_ts)
                base = rng.lognormal(0.0, 0.4)
                lift = effect if (arm == "T" and store_triggered) else 0.0
                add(uid, arm, store, "revenue", base * (1.0 + lift), revenue_ts)
                add(uid, arm, store, "revenue", base * (1.0 + lift), freshness_ts)
    return pa.table(rows)


@pytest.mark.parametrize(
    "method_name",
    ("run_daily", "run_daily_lift", "run_asof", "run_asof_lift"),
)
def test_triggered_clustered_day_axis_keeps_existing_refusal(tmp_path, method_name):
    con = ibis.duckdb.connect()
    con.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(
            n_stores_per_arm=5,
            units_per_store=4,
            trigger_rate=1.0,
        ),
    )
    path = tmp_path / "clustered.yml"
    path.write_text(_defs_yaml_clustered())
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        with pytest.raises(CapabilityError) as raised:
            getattr(analysis, method_name)(population="triggered")
        assert raised.value.code == "facade.analysis.clustered_day_axis"
    finally:
        analysis.close()
        con.disconnect()


def test_clustered_experiment_with_trigger_narrows_to_the_triggered_population(tmp_path):
    """A clustered experiment routes its triggered readout through the
    source-owned triggered-population operation."""
    rate, effect = 0.5, 0.5
    con = ibis.duckdb.connect()
    con.create_table(
        "cluster_trigger_events",
        obj=_clustered_trigger_events(n_stores_per_arm=40, trigger_rate=rate, effect=effect),
    )
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml_clustered())
    an = Analysis.from_definitions(
        "exp",
        defs_path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    rows = _lift_rows(an.run())
    by_pop = {r.analysis_population: r for r in rows if r.metric == "revenue"}
    assert set(by_pop) == {"assigned", "triggered"}
    assigned, triggered = by_pop["assigned"], by_pop["triggered"]

    # Assigned lift is the effect diluted by the trigger rate; triggered lift
    # recovers it. Equal or reversed values would mean both read the assigned source.
    assert triggered.require_lift().value > assigned.require_lift().value
    assert triggered.require_lift().value == pytest.approx(effect, rel=0.3)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_staggered_cluster_trigger_windows_censor_by_each_member_anchor(tmp_path):
    definitions = _defs_yaml_clustered().replace(
        "    start: 2024-01-01T00:00:00",
        "    start: 2024-01-01T00:00:00\n"
        "    end: 2024-01-01T00:00:00\n"
        "    observation_end: 2024-01-20T00:00:00",
    )
    path = tmp_path / "staggered-clusters.yml"
    path.write_text(definitions)
    rows = []
    for arm in ("C", "T"):
        for timing, trigger_day, revenue_day, value in (
            ("early", 2, 3, 1.0),
            ("late", 8, 9, 2.0),
        ):
            unit_id = f"{arm}-{timing}"
            cluster_id = f"{arm}-store"
            rows.extend(
                [
                    {
                        "user_id": unit_id,
                        "group_id": arm,
                        "store_id": cluster_id,
                        "event": "enrolled",
                        "ts": np.datetime64("2024-01-01T00:00:00"),
                        "value": 0.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": unit_id,
                        "group_id": arm,
                        "store_id": cluster_id,
                        "event": "saw_surface",
                        "ts": np.datetime64(f"2024-01-{trigger_day:02d}T00:00:00"),
                        "value": 0.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": unit_id,
                        "group_id": arm,
                        "store_id": cluster_id,
                        "event": "revenue",
                        "ts": np.datetime64(f"2024-01-{revenue_day:02d}T00:00:00"),
                        "value": value,
                        "experiment_id": "exp",
                    },
                ]
            )
    con = ibis.duckdb.connect()
    con.create_table("cluster_trigger_events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 12, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 12, tzinfo=dt.UTC)},
        ),
    )
    try:
        from tests.analysis_factory import _native_source

        with pytest.warns(IncrementWarning):
            frame = (
                _native_source(analysis)
                .unit_frame(analysis.metrics[0], population="triggered")
                .to_pylist()
            )
        assert {row["unit_id"] for row in frame} == {"C-early", "T-early"}
        assert {row["cluster_id"] for row in frame} == {"C-store", "T-store"}
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_asof_uses_per_unit_entry_and_maturity_without_future_members(
    tmp_path,
):
    rows = []
    trigger_days = {
        "C-early": 2,
        "C-late": 8,
        "T-early": 5,
        "T-late": 8,
    }
    for unit, trigger_day in trigger_days.items():
        arm = unit[0]
        rows.extend(
            [
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64("2024-01-01T00:00:00"),
                    "event": "enrolled",
                    "value": 0.0,
                    "experiment_id": "exp",
                },
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64(f"2024-01-{trigger_day:02d}T00:00:00"),
                    "event": "saw_surface",
                    "value": 0.0,
                    "experiment_id": "exp",
                },
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64(f"2024-01-{trigger_day + 1:02d}T00:00:00"),
                    "event": "revenue",
                    "value": 1.0 if arm == "C" else 2.0,
                    "experiment_id": "exp",
                },
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64("2024-01-16T00:00:00"),
                    "event": "revenue",
                    "value": 0.0,
                    "experiment_id": "exp",
                },
            ]
        )
    analysis = _analysis(tmp_path, pa.Table.from_pylist(rows))
    try:
        asof = analysis.run_asof(population="triggered")
        by_day_arm = {(row.ds.isoformat(), row.group_id): row for row in asof}
        assert by_day_arm[("2024-01-04", "C")].n == 1
        assert by_day_arm[("2024-01-04", "T")].n == 0
        assert by_day_arm[("2024-01-05", "T")].n == 1
        assert by_day_arm[("2024-01-07", "T")].n == 1
        assert by_day_arm[("2024-01-08", "C")].n == 2
        assert by_day_arm[("2024-01-08", "T")].n == 2

        completed = analysis.run_asof(
            population="triggered",
            completed_windows_only=True,
        )
        mature = {(row.ds.isoformat(), row.group_id): row.n for row in completed}
        assert mature[("2024-01-14", "C")] == 1
        assert mature[("2024-01-14", "T")] == 1
        assert mature[("2024-01-15", "C")] == 2
        assert mature[("2024-01-15", "T")] == 2

        lift = analysis.run_asof_lift(population="triggered")
        early_treatment = [
            row for row in lift if row.ds.isoformat() == "2024-01-04" and row.group_id == "T"
        ]
        assert early_treatment
        assert all(row.lift is None and row.unavailable == "few_units" for row in early_treatment)
        assert {row.failure_code for row in early_treatment} == {"readout.cell.missing_arm"}
        assert {row.sampling_reason_code for row in early_treatment} == {"readout.cell.missing_arm"}
        failed_cells = [
            cell
            for cell in lift.metadata.cells
            if cell.cell.group_id == "T" and cell.failure is not None
        ]
        assert failed_cells
        assert {cell.failure.code for cell in failed_cells} == {
            "readout.cell.missing_arm",
            "readout.cell.missing_metric_observations",
        }
        frame = lift.to_frame(backend="pyarrow").to_pylist()
        early_rows = [
            row
            for row in frame
            if row["group_id"] == "T" and row["failure_code"] == "readout.cell.missing_arm"
        ]
        assert early_rows
        from increment.estimation.readout_types import ReadoutResults

        restored = ReadoutResults.model_validate_json(lift.model_dump_json())
        assert {
            cell.failure.code
            for cell in restored.metadata.cells
            if cell.cell.group_id == "T" and cell.failure is not None
        } == {
            "readout.cell.missing_arm",
            "readout.cell.missing_metric_observations",
        }

    finally:
        analysis.close()


@pytest.mark.parametrize(
    "method_name",
    ("run_daily", "run_daily_lift", "run_asof", "run_asof_lift"),
)
def test_triggered_day_axis_emits_typed_cells_without_eligible_trigger_rows(tmp_path, method_name):
    events = _events(n_per_arm=2).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 17)
    analysis = _analysis(tmp_path, pa.Table.from_pylist(events))
    try:
        rows = getattr(analysis, method_name)(population="triggered")
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.scope.populations == ("triggered",)
        assert {row.analysis_population for row in rows} == {"triggered"}
        assert {row.ds for row in rows} == {dt.date(2024, 1, day) for day in range(1, 16)}
        assert {row.group_id for row in rows} == {"C", "T"}
        if method_name.endswith("_lift"):
            codes = {cell.failure.code for cell in rows.metadata.cells if cell.failure is not None}
            assert codes <= {
                "readout.cell.missing_arm",
                "readout.cell.missing_metric_observations",
            }
            assert "readout.cell.missing_arm" in codes
            assert all(row.failure_code in codes for row in rows if row.failure_code)
        else:
            assert all(row.value is None and row.n == 0 for row in rows)
            assert all(row.decision_scope_complete is False for row in rows)
    finally:
        analysis.close()


@pytest.mark.parametrize("method_name", ("run_asof", "run_asof_lift"))
def test_completed_triggered_windows_emit_cells_when_no_units_are_mature(tmp_path, method_name):
    events = _events(n_per_arm=2).to_pylist()
    for row in events:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 15)
    analysis = _analysis(tmp_path, pa.Table.from_pylist(events))
    try:
        rows = getattr(analysis, method_name)(population="triggered", completed_windows_only=True)
        assert rows
        assert rows.metadata.scope.populations == ("triggered",)
        assert {row.analysis_population for row in rows} == {"triggered"}
        assert {row.ds for row in rows} == {dt.date(2024, 1, day) for day in range(1, 16)}
        assert {row.group_id for row in rows} == {"C", "T"}
        if method_name.endswith("_lift"):
            codes = {cell.failure.code for cell in rows.metadata.cells if cell.failure is not None}
            assert codes <= {
                "readout.cell.missing_arm",
                "readout.cell.missing_metric_observations",
            }
            assert "readout.cell.missing_arm" in codes
            assert all(row.failure_code in codes for row in rows if row.failure_code)
        else:
            assert all(row.value is None and row.n == 0 for row in rows)
            assert all(row.decision_scope_complete is False for row in rows)
    finally:
        analysis.close()


def test_triggered_dimension_fills_missing_arm_within_each_segment(tmp_path):
    definitions = (
        _defs_yaml()
        .replace(
            "        column: null\nexposures:",
            "        column: null\n"
            "    properties:\n"
            "      - name: country\n"
            "        column: country\n"
            "        as_of: static\n"
            "exposures:",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    breakouts:\n"
            "      - {property: country, source: events}\n"
            "    plan:\n"
            "      secondaries: [revenue]",
        )
    )
    path = tmp_path / "country-trigger.yml"
    path.write_text(definitions)
    rows = []
    for unit, arm, country, trigger_day in (
        ("C-US", "C", "US", 2),
        ("C-CA", "C", "CA", 2),
        ("T-US", "T", "US", 2),
        ("T-CA", "T", "CA", 8),
    ):
        for event, day, value in (
            ("enrolled", 1, 0.0),
            ("saw_surface", trigger_day, 0.0),
            ("revenue", trigger_day + 1, 3.0 if arm == "C" else 6.0),
            ("revenue", 16, 0.0),
        ):
            rows.append(
                {
                    "user_id": unit,
                    "group_id": arm,
                    "country": country,
                    "ts": np.datetime64(f"2024-01-{day:02d}T00:00:00"),
                    "event": event,
                    "value": value,
                    "experiment_id": "exp",
                }
            )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        estimates = analysis.run_asof_lift(dimension="country", population="triggered")
        california_treatment = [
            row
            for row in estimates
            if row.ds == dt.date(2024, 1, 3) and row.group_id == "T" and row.dimension_value == "CA"
        ]
        assert california_treatment
        assert {row.failure_code for row in california_treatment} == {"readout.cell.missing_arm"}
        assert all(
            row.sampling_reason_code == "readout.cell.missing_arm" for row in california_treatment
        )
    finally:
        analysis.close()
        con.disconnect()


def _country_analysis(
    tmp_path,
    unit_specs,
    *,
    cutoff_day=16,
    observation_end=None,
    end=None,
    view_correction=None,
    vary_outcome=False,
    second_source=False,
    conversion=False,
    retention=False,
):
    definitions = (
        _defs_yaml(end=end, observation_end=observation_end)
        .replace(
            "        column: null\nexposures:",
            "        column: null\n"
            "    properties:\n"
            "      - name: country\n"
            "        column: country\n"
            "        as_of: static\n"
            "exposures:",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    breakouts:\n"
            "      - {property: country, source: events}\n"
            "    plan:\n"
            "      secondaries: [revenue]",
        )
    )
    if second_source:
        definitions = definitions.replace(
            "exposures:\n",
            "  - name: events_secondary\n"
            "    sql: SELECT * FROM events\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts: []\n"
            "    properties:\n"
            "      - name: country\n"
            "        column: country\n"
            "        as_of: static\n"
            "exposures:\n",
        ).replace(
            "      - {property: country, source: events}\n",
            "      - {property: country, source: events}\n"
            "      - {property: country, source: events_secondary}\n",
        )
    if view_correction is not None:
        definitions = definitions.replace(
            "    plan:\n      secondaries: [revenue]",
            f"    plan:\n      view_multiplicity: {{correction: {view_correction}}}\n"
            "      secondaries: [revenue]",
        )
    if conversion:
        definitions = definitions.replace(
            "experiments:\n",
            "  - name: conversion\n"
            "    type: conversion\n"
            "    entity: user_id\n"
            "    fact: revenue\n"
            "    window_days: 7\n"
            "experiments:\n",
        )
        definitions = definitions.replace(
            "secondaries: [revenue]", "secondaries: [revenue, conversion]"
        )
    if retention:
        definitions = definitions.replace(
            "experiments:\n",
            "  - name: retention\n"
            "    type: retention\n"
            "    entity: user_id\n"
            "    fact: revenue\n"
            "    threshold_days: [0, 7]\n"
            "experiments:\n",
        )
        definitions = definitions.replace(
            "secondaries: [revenue]", "secondaries: [revenue, retention]"
        )
    path = tmp_path / "country-trigger.yml"
    path.write_text(definitions)
    rows = []
    for unit, arm, country, trigger_day in unit_specs:
        value = 3.0 if arm == "C" else 6.0
        if vary_outcome and unit.rsplit("-", 1)[-1].isdigit():
            value += 0.25 * int(unit.rsplit("-", 1)[-1])
        for event, day, event_value in (
            ("enrolled", 1, 0.0),
            ("saw_surface", trigger_day, 0.0),
            ("revenue", trigger_day + 1, value),
            ("revenue", cutoff_day, 0.0),
        ):
            rows.append(
                {
                    "user_id": unit,
                    "group_id": arm,
                    "country": country,
                    "ts": np.datetime64(f"2024-01-{day:02d}T00:00:00"),
                    "event": event,
                    "value": event_value,
                    "experiment_id": "exp",
                }
            )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, cutoff_day, tzinfo=dt.UTC),
            {
                "events": dt.datetime(2024, 1, cutoff_day, tzinfo=dt.UTC),
                **(
                    {"events_secondary": dt.datetime(2024, 1, cutoff_day, tzinfo=dt.UTC)}
                    if second_source
                    else {}
                ),
            },
        ),
    )
    return analysis, con


def test_triggered_dimension_includes_assigned_segments_with_no_eligible_members(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 2),
            ("T-US", "T", "US", 2),
            ("C-CA", "C", "CA", 17),
            ("T-CA", "T", "CA", 17),
        ),
    )
    try:
        values = analysis.run_daily(dimension="country", population="triggered")
        california = [row for row in values if row.dimension_value == "CA"]
        assert california
        assert all(row.value is None and row.n == 0 for row in california)
        lifts = analysis.run_daily_lift(dimension="country", population="triggered")
        california_lifts = [row for row in lifts if row.dimension_value == "CA"]
        assert california_lifts
        assert all(row.failure_code == "readout.cell.missing_arm" for row in california_lifts)
    finally:
        analysis.close()
        con.disconnect()


def test_empty_triggered_segment_rows_are_scoped_to_each_breakout_source(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (("C-US", "C", "US", 17), ("T-US", "T", "US", 17)),
        second_source=True,
    )
    try:
        rows = analysis.run_asof_lift(dimension="country", population="triggered")
        us_rows = [row for row in rows if row.dimension_value == "US"]
        assert us_rows
        assert {row.source for row in us_rows} == {"events", "events_secondary"}
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_asof_empty_segment_keeps_bonferroni_family_size(tmp_path):
    one_segment = (
        *((f"C-US-{i}", "C", "US", 2) for i in range(8)),
        *((f"T-US-{i}", "T", "US", 2) for i in range(8)),
    )
    two_segment = (
        *one_segment,
        ("C-CA", "C", "CA", 17),
        ("T-CA", "T", "CA", 17),
    )
    two_path = tmp_path / "two-segments"
    one_path = tmp_path / "one-segment"
    two_path.mkdir()
    one_path.mkdir()
    two, two_con = _country_analysis(
        two_path, two_segment, view_correction="bonferroni", vary_outcome=True
    )
    one, one_con = _country_analysis(
        one_path, one_segment, view_correction="bonferroni", vary_outcome=True
    )
    try:
        two_rows = two.run_asof_lift(dimension="country", population="triggered")
        one_rows = one.run_asof_lift(dimension="country", population="triggered")
        two_us = next(
            row
            for row in two_rows
            if row.ds == dt.date(2024, 1, 4) and row.group_id == "T" and row.dimension_value == "US"
        )
        one_us = next(
            row
            for row in one_rows
            if row.ds == dt.date(2024, 1, 4) and row.group_id == "T" and row.dimension_value == "US"
        )
        assert two_us.require_lift().alpha == pytest.approx(one_us.require_lift().alpha / 2)
        assert (
            two_us.require_lift().value,
            two_us.n_treat,
            two_us.n_control,
            two_us.alternative,
            two_us.null_lift,
            two_us.policy_name,
        ) == (
            one_us.require_lift().value,
            one_us.n_treat,
            one_us.n_control,
            one_us.alternative,
            one_us.null_lift,
            one_us.policy_name,
        )
    finally:
        two.close()
        one.close()
        two_con.disconnect()
        one_con.disconnect()


def test_triggered_daily_primary_keeps_full_treatment_roster_alpha_when_arm_is_empty(
    tmp_path,
):
    definitions = _defs_yaml().replace(
        "    plan:\n      secondaries: [revenue]",
        "    plan:\n      primary: revenue\n      alternative: greater",
    )
    path = tmp_path / "three-arm.yml"
    path.write_text(definitions)
    events = []
    for group in ("C", "T1", "T2"):
        for index in range(40):
            user_id = f"{group}-{index}"
            events.extend(
                (
                    {
                        "user_id": user_id,
                        "group_id": group,
                        "ts": dt.datetime(2024, 1, 1),
                        "event": "enrolled",
                        "value": 0.0,
                        "experiment_id": "exp",
                    },
                    {
                        "user_id": user_id,
                        "group_id": group,
                        "ts": dt.datetime(2024, 1, 2),
                        "event": "saw_surface",
                        "value": 0.0,
                        "experiment_id": "exp",
                    },
                )
            )
            if group != "T2":
                value = (7.0 if index % 2 == 0 else 9.0) + (0.4 if group == "T1" else 0.0)
                events.append(
                    {
                        "user_id": user_id,
                        "group_id": group,
                        "ts": dt.datetime(2024, 1, 3),
                        "event": "revenue",
                        "value": value,
                        "experiment_id": "exp",
                    }
                )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(events))
    cutoff = dt.datetime(2024, 1, 5, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            cutoff,
            {"events": cutoff},
        ),
    )
    try:
        rows = analysis.run_daily_lift(population="triggered")
        treatment = next(
            row for row in rows if row.ds == dt.date(2024, 1, 3) and row.group_id == "T1"
        )
        assert treatment.unavailable is None, treatment.model_dump()
        lift = treatment.require_lift()
        assert lift.log_mean is not None and lift.log_se is not None
        from scipy.stats import norm as _norm

        p_value = float(_norm.sf(lift.log_mean / lift.log_se))
        assert lift.alpha is not None
        assert lift.alpha == pytest.approx(0.05)
        assert lift.alpha / 2.0 == pytest.approx(0.025)
        assert 0.025 < p_value < 0.05
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_asof_empty_dimension_with_unbounded_retention_uses_asof_roster(
    tmp_path,
):
    from increment.semantics.models import RetentionMetric

    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 17),
            ("T-US", "T", "US", 17),
            ("C-CA", "C", "CA", 17),
            ("T-CA", "T", "CA", 17),
        ),
        cutoff_day=4,
    )
    retention = RetentionMetric(
        name="retention", entity="user_id", fact="revenue", threshold_days=7
    )
    try:
        rows = analysis.run_asof(metrics=[retention], dimension="country", population="triggered")
        assert rows
        assert {row.dimension_value for row in rows} == {"US", "CA"}
        assert all(row.value is None and row.n == 0 for row in rows)
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_empty_retention_cohort_cell_keeps_cohort_basis(tmp_path):
    from increment.semantics.models import RetentionMetric

    analysis, con = _country_analysis(
        tmp_path,
        (("C-US", "C", "US", 17), ("T-US", "T", "US", 17)),
        cutoff_day=4,
        retention=True,
    )
    retention = RetentionMetric(
        name="retention", entity="user_id", fact="revenue", threshold_days=(0, 7)
    )
    try:
        rows = analysis.run_daily_lift(
            metrics=[retention], dimension="country", population="triggered"
        )
        assert rows
        assert all(row.ds_basis == "cohort" for row in rows)
        assert all(row.failure_code == "readout.cell.missing_arm" for row in rows)
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_roster_uses_only_the_evidence_breakout_source():
    from types import SimpleNamespace

    from increment._day_axis import _assigned_segment_roster
    from increment.semantics.models import Breakout

    breakouts = (
        Breakout(property="country", source="first"),
        Breakout(property="country", source="second"),
    )
    calls = []

    class Source:
        def assigned_breakout_values(self, breakout, *, source_name=None):
            calls.append((breakout.source, source_name))
            if breakout.source == source_name:
                return {"first": ("US",), "second": ("CA",)}[breakout.source]
            return ()

    roster = _assigned_segment_roster(
        cast(Any, Source()),
        "country",
        cast(Any, SimpleNamespace(breakouts=breakouts)),
        "first",
    )
    assert roster == ["US"]
    assert calls == [("first", "first"), ("second", "first")]


def test_triggered_empty_dimension_respects_declared_observation_horizon(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (("C-US", "C", "US", 17), ("T-US", "T", "US", 17)),
        observation_end='"2024-01-04"',
        end='"2024-01-04"',
    )
    try:
        rows = analysis.run_daily(dimension="country", population="triggered")
        assert {row.ds for row in rows} == {dt.date(2024, 1, day) for day in range(1, 5)}
        assert {row.dimension_value for row in rows} == {"US"}
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_empty_dimensioned_slice_retains_assigned_segment_roster(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 17),
            ("T-US", "T", "US", 17),
            ("C-CA", "C", "CA", 17),
            ("T-CA", "T", "CA", 17),
        ),
    )
    try:
        for method_name in ("run_daily", "run_daily_lift"):
            rows = getattr(analysis, method_name)(dimension="country", population="triggered")
            assert rows
            assert {row.dimension_value for row in rows} == {"US", "CA"}
            assert {row.analysis_population for row in rows} == {"triggered"}
        lifts = analysis.run_daily_lift(dimension="country", population="triggered")
        assert lifts.metadata.scope.families
        assert {family.dimension for family in lifts.metadata.scope.families} == {"country"}
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_lift_marks_treatment_when_control_segment_is_unavailable(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 2),
            ("T-US", "T", "US", 2),
            ("C-CA", "C", "CA", 17),
            ("T-CA-1", "T", "CA", 2),
            ("T-CA-2", "T", "CA", 2),
        ),
    )
    try:
        estimates = analysis.run_daily_lift(dimension="country", population="triggered")
        treatment = [
            row
            for row in estimates
            if row.ds == dt.date(2024, 1, 3) and row.group_id == "T" and row.dimension_value == "CA"
        ]
        assert treatment
        assert all(row.lift is None and row.unavailable == "no_control_arm" for row in treatment)
        assert {row.failure_code for row in treatment} == {"readout.cell.missing_arm"}
        assert {row.sampling_reason_code for row in treatment} == {"readout.cell.missing_arm"}
        cell = next(
            cell
            for cell in estimates.metadata.cells
            if cell.cell.group_id == "T"
            and cell.cell.ds == dt.date(2024, 1, 3)
            and cell.cell.dimension_value == "CA"
        )
        assert cell.failure.code == "readout.cell.missing_arm"
        from increment.estimation.readout_types import ReadoutResults

        restored = ReadoutResults.model_validate_json(estimates.model_dump_json())
        restored_cell = next(
            cell
            for cell in restored.metadata.cells
            if cell.cell.group_id == "T"
            and cell.cell.ds == dt.date(2024, 1, 3)
            and cell.cell.dimension_value == "CA"
        )
        assert restored_cell.failure == cell.failure
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_missing_lift_rows_use_each_groups_observed_count(tmp_path):
    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 2),
            ("C-US-2", "C", "US", 2),
            ("T-US", "T", "US", 17),
        ),
        cutoff_day=4,
    )
    try:
        estimates = analysis.run_daily_lift(dimension="country", population="triggered")
        assert all(
            row.failure_code is None
            for row in estimates
            if row.group_id == "C" and row.ds == dt.date(2024, 1, 3) and row.dimension_value == "US"
        )
        control_cells = [
            cell
            for cell in estimates.metadata.cells
            if cell.cell.group_id == "C" and cell.cell.ds == dt.date(2024, 1, 3)
        ]
        assert all(cell.failure is None for cell in control_cells)
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_empty_conversion_rows_are_typed_before_estimation(tmp_path):
    from increment.semantics.models import ConversionMetric

    analysis, con = _country_analysis(
        tmp_path,
        (
            ("C-US", "C", "US", 17),
            ("T-US", "T", "US", 17),
        ),
        cutoff_day=4,
    )
    metric = ConversionMetric(name="conversion", entity="user_id", fact="revenue", window_days=7)
    try:
        rows = analysis.run_daily_lift(metrics=[metric], population="triggered")
        assert rows
        assert {row.failure_code for row in rows} == {"readout.cell.missing_arm"}
        assert all(row.sampling_available is False for row in rows)
    finally:
        analysis.close()
        con.disconnect()


def test_triggered_conversion_mixed_empty_days_and_segments_skip_empty_fits(tmp_path):
    from increment.semantics.models import ConversionMetric

    populated = (
        *((f"C-US-{i}", "C", "US", 2) for i in range(8)),
        *((f"T-US-{i}", "T", "US", 2) for i in range(8)),
    )
    with_empty = (*populated, ("C-CA", "C", "CA", 17), ("T-CA", "T", "CA", 17))
    baseline_path = tmp_path / "populated"
    mixed_path = tmp_path / "mixed"
    baseline_path.mkdir()
    mixed_path.mkdir()
    baseline, baseline_con = _country_analysis(
        baseline_path, populated, cutoff_day=4, vary_outcome=True, conversion=True
    )
    mixed, mixed_con = _country_analysis(
        mixed_path, with_empty, cutoff_day=4, vary_outcome=True, conversion=True
    )
    metric = ConversionMetric(name="conversion", entity="user_id", fact="revenue", window_days=7)
    try:
        baseline_rows = baseline.run_daily_lift(
            metrics=[metric], dimension="country", population="triggered"
        )
        mixed_rows = mixed.run_daily_lift(
            metrics=[metric], dimension="country", population="triggered"
        )
        baseline_us = {
            (row.ds, row.group_id): row.lift.value
            for row in baseline_rows
            if row.dimension_value == "US" and row.lift is not None
        }
        mixed_us = {
            (row.ds, row.group_id): row.lift.value
            for row in mixed_rows
            if row.dimension_value == "US" and row.lift is not None
        }
        empty_ca = [row for row in mixed_rows if row.dimension_value == "CA"]
        assert baseline_us
        assert mixed_us.keys() == baseline_us.keys()
        assert mixed_us == pytest.approx(baseline_us)
        assert empty_ca
        assert all(row.failure_code == "readout.cell.missing_arm" for row in empty_ca)
        assert all(row.sampling_available is False for row in empty_ca)
    finally:
        baseline.close()
        mixed.close()
        baseline_con.disconnect()
        mixed_con.disconnect()


def test_triggered_single_unit_exact_binomial_cells_keep_sampling_evidence(tmp_path):
    definitions = (
        _defs_yaml()
        .replace(
            "  - name: revenue\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: revenue\n"
            "    aggregation: sum\n"
            "    window_days: 7",
            "  - name: converted\n"
            "    type: conversion\n"
            "    entity: user_id\n"
            "    fact: converted\n"
            "    window_days: 7",
        )
        .replace(
            "      - name: revenue\n",
            "      - name: converted\n        column: null\n      - name: revenue\n",
        )
        .replace("      secondaries: [revenue]", "      secondaries: [converted]")
    )
    path = tmp_path / "exact-binomial-trigger.yml"
    path.write_text(definitions)
    rows = []
    for unit, arm, converted in (("C1", "C", True), ("T1", "T", True)):
        rows.extend(
            [
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64("2024-01-01T00:00:00"),
                    "event": "enrolled",
                    "value": 0.0,
                    "experiment_id": "exp",
                },
                {
                    "user_id": unit,
                    "group_id": arm,
                    "ts": np.datetime64("2024-01-01T00:01:00"),
                    "event": "saw_surface",
                    "value": 0.0,
                    "experiment_id": "exp",
                },
                *(
                    [
                        {
                            "user_id": unit,
                            "group_id": arm,
                            "ts": np.datetime64("2024-01-03T00:00:00"),
                            "event": "converted",
                            "value": 1.0,
                            "experiment_id": "exp",
                        }
                    ]
                    if converted
                    else []
                ),
            ]
        )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        estimates = analysis.run_daily_lift(metrics=["converted"], population="triggered")
        treatment = [row for row in estimates if row.group_id == "T"]
        assert treatment
        assert all(row.reference_kind == "binomial" for row in treatment)
        assert all(row.binomial_set is not None for row in treatment)
        assert all(row.sampling_available for row in treatment)
        assert all(row.failure_code is None for row in treatment)
    finally:
        analysis.close()
        con.disconnect()


def _defs_yaml_quantile(trigger: str | None = "saw_surface") -> str:
    trigger_line = f"    trigger: {trigger}\n" if trigger else ""
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM quantile_trigger_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: latency
        column: value
      - name: enrolled
        column: null
      - name: saw_surface
        column: null
exposures:
  - name: assignment
    fact: enrolled
  - name: saw_surface
    fact: saw_surface
metrics:
  - name: p90_latency
    type: quantile
    quantile: 0.9
    entity: user_id
    fact: latency
    aggregation: sum
experiments:
  - name: exp
    exposure: assignment
{trigger_line}    unit: user_id
    start: 2024-01-01T00:00:00
    control_group: C
    plan:
      secondaries: [p90_latency]
"""


def _quantile_trigger_events(n_per_arm=300, trigger_rate=0.3, shift=1.6, seed=7):
    """Only triggered treatment units see the latency shift, mirroring
    `_events`'s dilution shape for the quantile per-unit path."""

    rng = np.random.default_rng(seed)
    rows: dict[str, list] = {
        "user_id": [],
        "group_id": [],
        "ts": [],
        "event": [],
        "value": [],
        "experiment_id": [],
    }
    enrolled_ts = np.datetime64("2024-01-02T00:00:00")
    metric_ts = np.datetime64("2024-01-02T01:00:00")

    def add(uid, arm, event, value, ts):
        rows["user_id"].append(uid)
        rows["group_id"].append(arm)
        rows["ts"].append(ts)
        rows["event"].append(event)
        rows["value"].append(value)
        rows["experiment_id"].append("exp")

    for arm in ("C", "T"):
        for i in range(n_per_arm):
            uid = f"{arm}{i}"
            triggered = rng.random() < trigger_rate
            add(uid, arm, "enrolled", 0.0, enrolled_ts)
            if triggered:
                add(uid, arm, "saw_surface", 0.0, enrolled_ts)
            mult = shift if (arm == "T" and triggered) else 1.0
            add(uid, arm, "latency", rng.lognormal(0.0, 0.5) * mult, metric_ts)
    return pa.table(rows)


def test_quantile_experiment_with_trigger_narrows_to_the_triggered_population(tmp_path):
    """A quantile experiment routes its triggered unit-frame readout through
    the source-owned triggered-population operation."""
    con = ibis.duckdb.connect()
    con.create_table("quantile_trigger_events", obj=_quantile_trigger_events())
    defs_path = tmp_path / "defs.yml"
    defs_path.write_text(_defs_yaml_quantile())
    an = Analysis.from_definitions(
        "exp",
        defs_path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    rows = _lift_rows(an.run())
    by_pop = {r.analysis_population: r for r in rows if r.metric == "p90_latency"}
    assert set(by_pop) == {"assigned", "triggered"}
    assigned, triggered = by_pop["assigned"], by_pop["triggered"]

    # Public assigned/triggered rows prove the native routing.
    # A triggered-only shift diluted by the trigger rate in the assigned
    # reading, undiluted in the triggered one -- equal values would mean
    # both were read off the same (assigned) source.
    assert triggered.require_lift().value > assigned.require_lift().value
