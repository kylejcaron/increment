from __future__ import annotations

import copy
import datetime as dt
import math
import pickle
from typing import Any, cast

import ibis
import pyarrow as pa
import pytest
from scipy.stats import norm

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import CapabilityError, CodedError, DefinitionError
from increment.estimation.sitewide import SitewideImpact, SitewideRatioImpact
from increment.query.artifact_contract import ArtifactContractError
from tests.analysis_factory import _moment_source as _arm_moment_source
from tests.analysis_factory import _native_source, native_connection
from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region


def _moment_source(analysis: Analysis) -> Any:
    """The analysis's source, for evidence operations beyond the moment protocol."""
    return _arm_moment_source(analysis)


def _analysis(
    tmp_path,
    events=None,
    *,
    store="auto",
    day_boundary=None,
    trigger=True,
    third_arm=False,
    watermark=dt.datetime(2024, 1, 17, tzinfo=dt.UTC),
    observation_end=None,
    source_snapshot_evidence=True,
    on_mixed_assignment="error",
    metric_shape="sum",
    observational=False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definitions = _defs_yaml(
        trigger="saw_surface" if trigger else None,
        end="2024-01-10",
        observation_end=observation_end,
    )
    if metric_shape != "windowed":
        definitions = definitions.replace("    window_days: 7\n", "")
    if metric_shape == "ratio_avg_event":
        definitions = definitions.replace(
            "  - name: revenue\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: revenue\n"
            "    aggregation: sum\n",
            "  - name: revenue_ratio\n"
            "    type: ratio\n"
            "    entity: user_id\n"
            "    numerator: {fact: revenue, aggregation: sum}\n"
            "    denominator: {fact: revenue, aggregation: avg_event}\n",
        ).replace("secondaries: [revenue]", "secondaries: [revenue_ratio]")
    elif metric_shape == "avg_event":
        definitions = definitions.replace("    aggregation: sum\n", "    aggregation: avg_event\n")
    if observational:
        definitions = definitions.replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        ).replace(
            "    plan:\n",
            "    design:\n"
            "      mechanism: observational\n"
            "      covariates:\n"
            "        - {property: region, source: events}\n"
            "    plan:\n",
        )
    if day_boundary is not None:
        definitions = definitions.replace(
            "    start: 2024-01-01T00:00:00\n",
            f"    day_boundary: {day_boundary}\n    start: 2024-01-01T00:00:00\n",
        )
    if third_arm:
        definitions = definitions.replace(
            "    control_group: C\n",
            "    control_group: C\n"
            "    allocation: {C: 0.4, T: 0.4, U: 0.2}\n"
            "    allocation_scheme: independent\n",
        )
    definitions_path = tmp_path / "triggered-sitewide.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    source_events = (
        _events(n_per_arm=100, trigger_rate=0.2, effect=0.5) if events is None else events
    )
    if observational:
        source_events = _with_preassignment_region(source_events)
    con.create_table("events", obj=source_events)
    return Analysis.from_definitions(
        "exp",
        definitions_path,
        con,
        store=store,
        source_snapshot_evidence=(
            SourceSnapshotEvidence(
                observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
                complete_through_by_feed={"events": watermark},
            )
            if source_snapshot_evidence
            else None
        ),
        on_mixed_assignment=on_mixed_assignment,
    )


def _conversion_analysis(tmp_path, events, *, count_metric=False):
    import yaml

    tmp_path.mkdir(parents=True, exist_ok=True)
    definitions = _defs_yaml(end="2024-01-10").replace("    window_days: 7\n", "")
    definitions = (
        definitions.replace("    type: mean\n", "    type: conversion\n")
        .replace("    fact: revenue\n", "    fact: converted\n")
        .replace("    aggregation: sum\n", "")
        .replace(
            "      - name: saw_surface\n        column: null\n",
            "      - name: saw_surface\n        column: null\n"
            "      - name: converted\n        column: null\n",
        )
    )
    if count_metric:
        payload = yaml.safe_load(definitions)
        payload["metrics"].insert(
            0,
            {
                "name": "aaa_count",
                "type": "mean",
                "entity": "user_id",
                "fact": "converted",
                "aggregation": "count",
            },
        )
        payload["experiments"][0]["plan"]["secondaries"].append("aaa_count")
        definitions = yaml.safe_dump(payload, sort_keys=False)
    definitions_path = tmp_path / "conversion-sitewide.yml"
    definitions_path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=events)
    return Analysis.from_definitions(
        "exp",
        definitions_path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 17, tzinfo=dt.UTC)},
        ),
    )


def _add_late_trigger_and_outcome(events, *, include_outcome):
    rows = events.to_pylist()
    triggered = {row["user_id"] for row in rows if row["event"] == "saw_surface"}
    unit_id = next(f"C{index}" for index in range(100) if f"C{index}" not in triggered)
    extra = [
        {
            "user_id": unit_id,
            "group_id": "C",
            "ts": dt.datetime(2024, 1, 12),
            "event": "saw_surface",
            "value": 0.0,
            "experiment_id": "exp",
        }
    ]
    if include_outcome:
        extra.append(
            {
                "user_id": unit_id,
                "group_id": "C",
                "ts": dt.datetime(2024, 1, 12),
                "event": "revenue",
                "value": 1_000_000.0,
                "experiment_id": "exp",
            }
        )
    return pa.concat_tables([events, pa.Table.from_pylist(extra, schema=events.schema)])


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.slow
def test_triggered_sitewide_evidence_uses_calendar_window_member_outcomes(tmp_path):
    base_events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    member_only_events = _add_late_trigger_and_outcome(base_events, include_outcome=False)
    with_late_outcome_events = _add_late_trigger_and_outcome(base_events, include_outcome=True)
    member_only = _analysis(
        tmp_path / "member-only",
        member_only_events,
        observation_end="2024-01-16",
    )
    with_late_outcome = _analysis(
        tmp_path / "late-outcome",
        with_late_outcome_events,
        observation_end="2024-01-16",
    )
    adopted = None
    try:
        source = _moment_source(with_late_outcome)
        evidence = source.sitewide_evidence(
            with_late_outcome.metrics[0], include_ratio=False, population="triggered"
        )
        calendar_start = dt.datetime(2024, 1, 1)
        calendar_end = dt.datetime(2024, 1, 10, 23, 59, 59, 999999)
        rows = with_late_outcome_events.to_pylist()
        members = {
            row["user_id"]
            for row in rows
            if row["event"] == "saw_surface" and row["ts"] <= dt.datetime(2024, 1, 16)
        }
        groups = {row["user_id"]: row["group_id"] for row in rows}
        sums: dict[str, float] = {}
        for row in rows:
            if row["event"] == "revenue" and calendar_start <= row["ts"] <= calendar_end:
                sums[row["user_id"]] = sums.get(row["user_id"], 0.0) + row["value"]
        expected_by_arm: dict[str, list[float]] = {}
        for unit in members:
            expected_by_arm.setdefault(groups[unit], []).append(sums.get(unit, 0.0))
        hand_computed_means = {
            group: sum(values) / len(values) for group, values in expected_by_arm.items()
        }
        hand_computed_site_total = sum(
            row["value"]
            for row in rows
            if row["event"] == "revenue" and calendar_start <= row["ts"] <= calendar_end
        )
        assert evidence.analysis_population == "triggered"
        assert evidence.window == (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert evidence.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert evidence.population_counts == source.assignment_counts(population="triggered")
        assert evidence.assigned_counts == {"C": 100, "T": 100}
        assert evidence.population_counts["C"] == 17
        assert evidence.site_total == pytest.approx(hand_computed_site_total)
        assert {arm.group_id: arm.n for arm in evidence.arm_stats} == {
            group: len(values) for group, values in expected_by_arm.items()
        }
        assert {arm.group_id: arm.mean_y() for arm in evidence.arm_stats} == pytest.approx(
            hand_computed_means
        )
        native_result = with_late_outcome.sitewide("revenue", population="triggered")
        reference_result = member_only.sitewide("revenue", population="triggered")
        adopted = _publish_and_adopt(
            native_connection(_native_source(with_late_outcome)),
            with_late_outcome,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact_result = adopted.sitewide("revenue", population="triggered")
        artifact_evidence = _moment_source(adopted).sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="triggered"
        )
        assert {arm.group_id: arm.n for arm in artifact_evidence.arm_stats} == {
            group: len(values) for group, values in expected_by_arm.items()
        }
        assert {arm.group_id: arm.mean_y() for arm in artifact_evidence.arm_stats} == pytest.approx(
            hand_computed_means
        )
        assert artifact_result.window == native_result.window == reference_result.window
        assert artifact_result.absolute_impact == pytest.approx(reference_result.absolute_impact)
        assert isinstance(artifact_result, SitewideImpact)
        assert isinstance(reference_result, SitewideImpact)
        assert artifact_result.baseline_volume == pytest.approx(reference_result.baseline_volume)
    finally:
        member_only.close()
        if adopted is None:
            with_late_outcome.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.parametrize("store", ["auto", "always", "none"])
@pytest.mark.parametrize(
    "order",
    [
        ("assigned", "triggered"),
        ("triggered", "triggered"),
        ("triggered", "assigned"),
    ],
)
def test_triggered_sitewide_reductions_do_not_cross_contaminate_cache(tmp_path, store, order):
    analysis = _analysis(tmp_path / "cached", store=store)
    fresh_triggered = _analysis(tmp_path / "fresh-triggered", store=store)
    fresh_assigned = _analysis(tmp_path / "fresh-assigned", store=store)
    try:
        expected = {
            "triggered": fresh_triggered.sitewide("revenue", population="triggered"),
            "assigned": fresh_assigned.sitewide("revenue"),
        }
        actual = {}
        for population in order:
            actual[population] = analysis.sitewide(
                "revenue",
                population="triggered" if population == "triggered" else "assigned",
            )
        for population, result in actual.items():
            reference = expected[population]
            assert result.delta == pytest.approx(reference.delta)
            assert result.n_control == pytest.approx(reference.n_control)
            assert result.n_treatment == pytest.approx(reference.n_treatment)
            assert result.analysis_population == reference.analysis_population
            assert result.absolute_impact == pytest.approx(reference.absolute_impact)
            assert result.baseline_volume == pytest.approx(reference.baseline_volume)
    finally:
        analysis.close()
        fresh_triggered.close()
        fresh_assigned.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_artifact_adoption_rejects_trigger_cutoff_mismatch(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "trigger_measure_stats"),
        )
        _con, store, reference = cast(Any, adopted)._parity_artifacts[0]
        with store.open_snapshot(reference) as snapshot:
            manifest = snapshot.read_manifest(
                reference.manifest, expected_sha256=reference.manifest_sha256
            )
        trigger = next(item for item in manifest.extensions if item.kind == "trigger_population")
        trigger_measure = next(
            item for item in manifest.extensions if item.kind == "trigger_measure_stats"
        )
        assert trigger.observation_cutoff_ts == trigger_measure.observation_cutoff_ts

        payload = manifest.model_dump(mode="python")
        for extension in payload["extensions"]:
            if extension["kind"] == "trigger_measure_stats":
                extension["observation_cutoff_ts"] = dt.datetime(2024, 1, 15, tzinfo=dt.UTC)
        with pytest.raises(DefinitionError) as refused:
            type(manifest).model_validate(payload)
        assert getattr(refused.value, "code", None) == "definition.unit_day.trigger_measure_anchor"
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_matches_native_evidence(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        native = _native_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = _moment_source(adopted).sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="triggered"
        )
        assert artifact.analysis_population == "triggered"
        assert artifact.population_counts == native.population_counts
        assert artifact.assigned_counts == native.assigned_counts
        assert artifact.site_total == pytest.approx(native.site_total, rel=1e-12)
        assert {arm.group_id: arm.n for arm in artifact.arm_stats} == {
            arm.group_id: arm.n for arm in native.arm_stats
        }
        assert {arm.group_id: arm.mean_y() for arm in artifact.arm_stats} == pytest.approx(
            {arm.group_id: arm.mean_y() for arm in native.arm_stats}, rel=1e-12
        )
        artifact_assigned = _moment_source(adopted).sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="assigned"
        )
        assert artifact.snapshot_id == artifact_assigned.snapshot_id
        assert artifact_assigned.analysis_population == "assigned"
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_refuses_incomplete_calendar_outcomes(tmp_path):
    analysis = _analysis(tmp_path, watermark=dt.datetime(2024, 1, 8, tzinfo=dt.UTC))
    adopted = None
    try:
        for population in ("assigned", "triggered"):
            with pytest.raises(CapabilityError) as native_refused:
                analysis.sitewide("revenue", population=population)
            assert native_refused.value.code == "facade.analysis.sitewide_outcomes_incomplete"
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        for population in ("assigned", "triggered"):
            with pytest.raises(CapabilityError) as artifact_refused:
                adopted.sitewide("revenue", population=population)
            assert artifact_refused.value.code == "facade.analysis.sitewide_outcomes_incomplete"
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_assigned_sitewide_uses_experiment_end_for_native_and_artifact(tmp_path):
    events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    late = pa.Table.from_pylist(
        [
            {
                **row,
                "ts": dt.datetime(2024, 1, 12),
                "value": 10_000.0,
            }
            for row in events.to_pylist()
            if row["event"] == "revenue" and row["user_id"] == "T0"
        ],
        schema=events.schema,
    )
    events = pa.concat_tables([events, late])
    analysis = _analysis(
        tmp_path,
        events,
        observation_end="2024-01-16",
    )
    adopted = None
    try:
        native = analysis.sitewide("revenue")
        assert isinstance(native, SitewideImpact)
        calendar_outcomes = [
            row
            for row in events.to_pylist()
            if row["event"] == "revenue" and row["ts"].date() <= dt.date(2024, 1, 10)
        ]
        by_arm = {
            arm: sum(row["value"] for row in calendar_outcomes if row["group_id"] == arm)
            for arm in ("C", "T")
        }
        expected_delta = by_arm["T"] / 100 - by_arm["C"] / 100
        expected_site_total = sum(row["value"] for row in calendar_outcomes)
        assert native.delta == pytest.approx(expected_delta)
        assert native.site_total_volume == pytest.approx(expected_site_total)
        assert native.absolute_impact == pytest.approx(expected_delta * 200)
        assert native.baseline_volume == pytest.approx(expected_site_total - expected_delta * 100)
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = adopted.sitewide("revenue")
        assert isinstance(artifact, SitewideImpact)
        assert artifact.delta == pytest.approx(expected_delta)
        assert artifact.site_total_volume == pytest.approx(expected_site_total)
        assert artifact.absolute_impact == pytest.approx(expected_delta * 200)
        assert artifact.baseline_volume == pytest.approx(expected_site_total - expected_delta * 100)
        assert artifact.delta == pytest.approx(native.delta)
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_missing_outcome_watermark_refuses_triggered_but_not_assigned_sitewide(tmp_path):
    # The cutoff covers the experiment end, but no feed watermark certifies it.
    analysis = _analysis(tmp_path, watermark=None)
    adopted = None
    try:
        native_assigned = analysis.sitewide("revenue")
        with pytest.raises(CapabilityError) as native_refused:
            analysis.sitewide("revenue", population="triggered")
        assert native_refused.value.code == "facade.analysis.sitewide_outcomes_incomplete"
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact_assigned = adopted.sitewide("revenue")
        assert isinstance(native_assigned, SitewideImpact)
        assert isinstance(artifact_assigned, SitewideImpact)
        assert artifact_assigned.delta == pytest.approx(native_assigned.delta)
        assert artifact_assigned.site_total_volume == pytest.approx(
            native_assigned.site_total_volume
        )
        assert artifact_assigned.absolute_impact == pytest.approx(native_assigned.absolute_impact)
        with pytest.raises(CapabilityError) as artifact_refused:
            adopted.sitewide("revenue", population="triggered")
        assert artifact_refused.value.code == native_refused.value.code
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_scales_complete_three_arm_roster(tmp_path):
    events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    third_arm_rows = []
    for index in range(20):
        unit, triggered = f"U{index}", index < 5
        third_arm_rows += [
            ("enrolled", 0.0, dt.datetime(2024, 1, 2), unit),
            *([("saw_surface", 0.0, dt.datetime(2024, 1, 2), unit)] if triggered else []),
            ("revenue", 1.0 + index / 10, dt.datetime(2024, 1, 8), unit),
        ]
    extra = pa.Table.from_pylist(
        [
            {
                "user_id": unit,
                "group_id": "U",
                "ts": ts,
                "event": event,
                "value": value,
                "experiment_id": "exp",
            }
            for event, value, ts, unit in third_arm_rows
        ],
        schema=events.schema,
    )
    analysis = _analysis(
        tmp_path,
        events=pa.concat_tables([events, extra]),
        third_arm=True,
    )
    adopted = None
    try:
        native = analysis.sitewide("revenue", arm="T", population="triggered")
        evidence = _moment_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        assert evidence.population_counts["U"] == 5
        assert native.scaled_units == round(native.n_enrolled)
        assert native.scaled_units == sum(
            arm.n for arm in evidence.arm_stats if arm.group_id in {"C", "T", "U"}
        )
        assert native.unobserved_members == 0
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = adopted.sitewide("revenue", arm="T", population="triggered")
        assert artifact.scaled_units == native.scaled_units
        assert artifact.unobserved_members == 0
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_native_sitewide_without_snapshot_evidence_has_no_snapshot_identity(tmp_path):
    analysis = _analysis(tmp_path, source_snapshot_evidence=False)
    try:
        assert analysis.sitewide("revenue").snapshot_id is None
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_sitewide_artifact_snapshot_identity_changes_between_publications(tmp_path):
    first = _analysis(tmp_path / "first")
    second = _analysis(tmp_path / "second")
    adopted_first = adopted_second = None
    try:
        adopted_first = _publish_and_adopt(
            native_connection(_native_source(first)),
            first,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        adopted_second = _publish_and_adopt(
            native_connection(_native_source(second)),
            second,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        assert (
            adopted_first.sitewide("revenue").snapshot_id
            != adopted_second.sitewide("revenue").snapshot_id
        )
    finally:
        for adopted in (adopted_first, adopted_second):
            if adopted is not None:
                _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_non_utc_boundary_matches_artifact(tmp_path):
    analysis = _analysis(tmp_path, day_boundary="UTC+05:00")
    adopted = None
    try:
        native = analysis.sitewide("revenue", population="triggered")
        assert native.window == (
            analysis.experiment.start_day,
            analysis.experiment.end_day,
        )
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = adopted.sitewide("revenue", population="triggered")
        assert artifact.window == native.window
        assert artifact.absolute_impact == pytest.approx(native.absolute_impact)
        assert isinstance(artifact, SitewideImpact)
        assert isinstance(native, SitewideImpact)
        assert artifact.baseline_volume == pytest.approx(native.baseline_volume)
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_without_members_refuses_alike_on_both_ingresses(tmp_path):
    analysis = _analysis(tmp_path, events=_events(n_per_arm=50, trigger_rate=0.0, effect=0.5))
    adopted = None
    try:
        with pytest.raises(CodedError) as native:
            analysis.sitewide("revenue", population="triggered")
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        with pytest.raises(CodedError) as artifact:
            adopted.sitewide("revenue", population="triggered")
        assert artifact.value.code == native.value.code
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_retains_assigned_arm_with_zero_members(tmp_path):
    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    for index in range(10):
        unit = f"U{index}"
        for event, value, timestamp in (
            ("enrolled", 0.0, dt.datetime(2024, 1, 2)),
            ("revenue", 1.0, dt.datetime(2024, 1, 8)),
        ):
            rows.append(
                {
                    "user_id": unit,
                    "group_id": "U",
                    "ts": timestamp,
                    "event": event,
                    "value": value,
                    "experiment_id": "exp",
                }
            )
    analysis = _analysis(
        tmp_path,
        pa.Table.from_pylist(rows),
        third_arm=True,
        watermark=dt.datetime(2024, 1, 12, tzinfo=dt.UTC),
    )
    adopted = None
    try:
        native_result = analysis.sitewide("revenue", population="triggered")
        native = _moment_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        assert native.assigned_counts == {"C": 100, "T": 100, "U": 10}
        assert native.population_counts.get("U") == 0
        assert "U" not in {arm.group_id for arm in native.arm_stats}
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact_result = adopted.sitewide("revenue", population="triggered")
        artifact = _moment_source(adopted).sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="triggered"
        )
        assert artifact.assigned_counts == native.assigned_counts
        assert artifact.population_counts == native.population_counts
        assert artifact.assigned_counts["U"] == 10
        assert artifact.population_counts["U"] == 0
        assert artifact_result.scaled_units == native_result.scaled_units
        assert artifact_result.absolute_impact == pytest.approx(native_result.absolute_impact)
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_balance_counts_assigned_arm_with_no_members(tmp_path):
    # No declared allocation: the balance check compares equal shares, and an
    # assigned arm nobody in which triggered must count as zero, not vanish.
    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    rows += [
        {
            "user_id": f"U{index}",
            "group_id": "U",
            "ts": dt.datetime(2024, 1, 2),
            "event": "enrolled",
            "value": 0.0,
            "experiment_id": "exp",
        }
        for index in range(100)
    ]
    analysis = _analysis(tmp_path, pa.Table.from_pylist(rows))
    adopted = None
    try:
        native = analysis.sitewide("revenue", arm="T", population="triggered")
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        artifact = adopted.sitewide("revenue", arm="T", population="triggered")
        for result in (native, artifact):
            assert result.allocation_balance is not None
            assert result.allocation_balance.status == "failed"
        assert artifact.allocation_balance == native.allocation_balance
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_without_membership_refuses_before_snapshot_read(
    tmp_path, monkeypatch
):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)), analysis, kinds=("site_volume",)
        )
        artifact_source = _moment_source(adopted)
        monkeypatch.setattr(
            type(artifact_source._snapshot),
            "execute",
            lambda *args, **kwargs: pytest.fail("read snapshot before required extension checks"),
        )
        with pytest.raises(ArtifactContractError) as raised:
            adopted.sitewide("revenue", population="triggered")
        for restored in (
            copy.deepcopy(raised.value),
            pickle.loads(pickle.dumps(raised.value)),
        ):
            assert restored.code == raised.value.code
            assert restored.context == raised.value.context
        assert raised.value.code == "artifact.extension.missing"
        assert "republish" in str(raised.value.context["route"])
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_without_assignment_counts_refuses(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population"),
        )
        with pytest.raises(ArtifactContractError) as refused:
            adopted.sitewide("revenue", population="triggered")
        assert refused.value.code == "artifact.extension.missing"
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_rejects_inconsistent_assignment_counts(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        source = _moment_source(adopted)
        original_read_extension = source._read_extension

        def inconsistent_counts(extension, *, request):
            rows = original_read_extension(extension, request=request)
            if extension.kind == "assignment_counts":
                rows = [dict(row) for row in rows]
                row = next(row for row in rows if row["population"] == "triggered")
                row["n_units"] = int(row["n_units"]) + 1
            return rows

        monkeypatch.setattr(source, "_read_extension", inconsistent_counts)
        with pytest.raises(ArtifactContractError) as refused:
            adopted.sitewide("revenue", population="triggered")
        assert refused.value.code == "artifact.extension.invalid"
        assert refused.value.context["extension_kind"] == "assignment_counts"
        assert "republish" in str(refused.value.context["route"])
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_rejects_corrupted_assignment_counts(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        con, store, reference = cast(Any, adopted)._parity_artifacts[0]
        with store.open_snapshot(reference) as snapshot:
            manifest = snapshot.read_manifest(
                reference.manifest, expected_sha256=reference.manifest_sha256
            )
            extension = next(
                item for item in manifest.extensions if item.kind == "assignment_counts"
            )
            rows = snapshot.execute(
                snapshot.verify_relation(extension.relation, expected_role="assignment_counts")
            ).to_pylist()
        rows[0]["n_units"] += 1
        from tests.test_unit_day_artifact_parity import _replace_artifact_relation

        _replace_artifact_relation(con, extension.relation, "assignment_counts", rows)
        with pytest.raises(ArtifactContractError) as refused:
            adopted.sitewide("revenue", population="triggered")
        assert refused.value.code == "artifact.snapshot.mixed"
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.parametrize(
    ("trigger", "source_snapshot_evidence", "code"),
    [
        (False, True, "source.native.operation"),
        (True, False, "source.native.trigger_evidence_required"),
    ],
)
def test_triggered_sitewide_evidence_refuses_before_resolving_exposures(
    tmp_path, monkeypatch, trigger, source_snapshot_evidence, code
):
    analysis = _analysis(
        tmp_path,
        trigger=trigger,
        source_snapshot_evidence=source_snapshot_evidence,
    )
    try:
        monkeypatch.setattr(
            _moment_source(analysis),
            "_get_exposures",
            lambda: pytest.fail("resolved exposures before validating trigger evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            _moment_source(analysis).sitewide_evidence(
                analysis.metrics[0], include_ratio=False, population="triggered"
            )
        assert refused.value.code == code
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("policy", ["error", "warn"])
def test_triggered_sitewide_evidence_applies_mixed_assignment_policy(tmp_path, policy):
    from increment.errors import IncrementWarning, InvalidRequestError

    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    assigned = next(row for row in rows if row["user_id"] == "C0" and row["event"] == "enrolled")
    events = pa.Table.from_pylist(
        [*rows, {**assigned, "group_id": "T"}],
        schema=pa.Table.from_pylist(rows).schema,
    )
    analysis = _analysis(tmp_path, events, on_mixed_assignment=policy)
    try:
        if policy == "error":
            with pytest.raises(InvalidRequestError) as refused:
                _moment_source(analysis).sitewide_evidence(
                    analysis.metrics[0], include_ratio=False, population="triggered"
                )
            assert refused.value.code == "query.integrity.mixed_assignment_units"
        else:
            with pytest.warns(IncrementWarning) as warning:
                evidence = _moment_source(analysis).sitewide_evidence(
                    analysis.metrics[0], include_ratio=False, population="triggered"
                )
            increment_warning = next(
                item.message for item in warning if isinstance(item.message, IncrementWarning)
            )
            assert increment_warning.code == "query.integrity.mixed_assignments_excluded"
            assert evidence.analysis_population == "triggered"
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_observational_design_refuses_before_evidence(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, observational=True)
    try:
        monkeypatch.setattr(
            _moment_source(analysis),
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("acquired observational sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            analysis.sitewide("revenue", population="triggered")
        assert refused.value.code == "facade.analysis.sitewide_requires_randomized"
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_assigned_sitewide_observational_design_refuses_before_evidence(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, observational=True)
    try:
        monkeypatch.setattr(
            _moment_source(analysis),
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("acquired observational sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            analysis.sitewide("revenue")
        assert refused.value.code == "facade.analysis.sitewide_requires_randomized"
    finally:
        analysis.close()


@pytest.mark.parametrize("population", ["assigned", "triggered"])
@pytest.mark.parametrize("ingress", ["native", "artifact"])
def test_windowed_sitewide_metric_refuses_before_data_load(
    tmp_path, monkeypatch, population, ingress
):
    from tests.capability_consumers_auxiliary import assert_refusal

    analysis = _analysis(tmp_path, metric_shape="windowed")
    adopted = None
    try:
        if ingress == "artifact":
            adopted = _publish_and_adopt(
                native_connection(_native_source(analysis)),
                analysis,
                kinds=("site_volume", "trigger_population", "assignment_counts"),
            )
            target = adopted
        else:
            target = analysis
        monkeypatch.setattr(
            _moment_source(target),
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("loaded incompatible sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            target.sitewide("revenue", population=population)
        assert refused.value.code == "facade.analysis.sitewide_metric_unaligned"
        assert refused.value.context["metric"] == "revenue"
        assert refused.value.context["metric_type"] == "mean"
        assert refused.value.context["window_days"] == 7
        ingress_name = "definitions" if ingress == "native" else "artifact"
        assert_refusal(
            "triggered_auxiliary_reads",
            f"sitewide_{ingress_name}_metric_unaligned_refused",
            refused.value,
            stage="read",
        )
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.parametrize("population", ["assigned", "triggered"])
@pytest.mark.parametrize("ingress", ["native", "artifact"])
@pytest.mark.parametrize("metric_kind", ["conversion", "avg_event_mean", "ratio"])
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_nonadditive_sitewide_metrics_refuse_before_evidence(
    tmp_path, monkeypatch, population, ingress, metric_kind
):
    if metric_kind == "conversion":
        target_source = _conversion_analysis(
            tmp_path, _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
        )
        metric_name = "revenue"
    else:
        target_source = _analysis(
            tmp_path, metric_shape="ratio_avg_event" if metric_kind == "ratio" else "avg_event"
        )
        metric_name = "revenue_ratio" if metric_kind == "ratio" else "revenue"

    adopted = None
    try:
        if ingress == "artifact":
            adopted = _publish_and_adopt(
                native_connection(_native_source(target_source)),
                target_source,
                kinds=("site_volume", "trigger_population", "assignment_counts"),
            )
            target = adopted
        else:
            target = target_source
        monkeypatch.setattr(
            _moment_source(target),
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("loaded incompatible sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            target.sitewide(metric_name, population=population)
        assert refused.value.code == "facade.analysis.sitewide_metric_unaligned"
        assert refused.value.context["metric"] == metric_name
        expected_type = (
            "conversion"
            if metric_kind == "conversion"
            else ("ratio" if metric_kind == "ratio" else "mean")
        )
        assert refused.value.context["metric_type"] == expected_type
    finally:
        if adopted is None:
            target_source.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_result_discloses_population_and_provenance(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        assigned = analysis.sitewide("revenue")
        result = analysis.sitewide("revenue", population="triggered")
        assert result.analysis_population == "triggered"
        assert result.window == (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert result.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert result.snapshot_id
        assert result.window == assigned.window
        assert result.snapshot_id == assigned.snapshot_id
        assert result.scaled_units == 33
        assert result.multiplicity_status == "exploratory_unadjusted"
        assert result.adjustment == "unadjusted"
        assert result.assumptions == (
            "membership_independent_of_treatment",
            "never_triggered_unaffected",
            "enrolled_units_exhaust_exposure",
        )
        assert result.allocation_balance is not None
        assert result.allocation_balance.status in {"passed", "failed"}
        assert "population='triggered'" in repr(result)
        assert "scaled_units=33" in repr(result)
        cls = type(result)
        assert cls.model_validate_json(result.model_dump_json()) == result
        assert pickle.loads(pickle.dumps(result)) == result
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_ratio_component_baseline(tmp_path):
    definitions = tmp_path / "ratio.yml"
    yaml = (
        _defs_yaml(end="2024-01-10")
        .replace(
            "      - name: saw_surface\n        column: null\n",
            "      - name: saw_surface\n        column: null\n"
            "      - name: session\n        column: value\n",
        )
        .replace(
            "  - name: revenue\n    type: mean\n    entity: user_id\n"
            "    fact: revenue\n    aggregation: sum\n    window_days: 7\n",
            "  - name: revenue_per_session\n    type: ratio\n    entity: user_id\n"
            "    numerator: {fact: revenue, aggregation: sum}\n"
            "    denominator: {fact: session, aggregation: sum}\n",
        )
    )
    yaml = yaml.replace("secondaries: [revenue]", "secondaries: [revenue_per_session]")
    definitions.write_text(yaml)
    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    triggered = {row["user_id"] for row in rows if row["event"] == "saw_surface"}
    for unit_id in {row["user_id"] for row in rows if row["event"] == "enrolled"}:
        value = 2.0 if unit_id.startswith("T") and unit_id in triggered else 1.0
        rows.append(
            {
                "user_id": unit_id,
                "group_id": unit_id[0],
                "ts": dt.datetime(2024, 1, 8),
                "event": "session",
                "value": value,
                "experiment_id": "exp",
            }
        )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )
    try:
        evidence = _moment_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=True, population="triggered"
        )
        result = analysis.sitewide("revenue_per_session", population="triggered")
        assert isinstance(result, SitewideRatioImpact)
        arms = {arm.group_id: arm for arm in evidence.arm_stats}
        control, treatment = arms["C"], arms["T"]
        delta_num = treatment.mean_y() - control.mean_y()
        delta_den = treatment.mean_den() - control.mean_den()
        n_t = treatment.n
        n_total = sum(evidence.population_counts.values())
        assert evidence.site_total_denominator is not None
        n0 = evidence.site_total - n_t * delta_num
        d0 = evidence.site_total_denominator - n_t * delta_den
        n1 = n0 + n_total * delta_num
        d1 = d0 + n_total * delta_den
        assert delta_den != pytest.approx(0)
        assert result.baseline_numerator == pytest.approx(n0, rel=1e-12)
        assert result.baseline_denominator == pytest.approx(d0, rel=1e-12)
        assert result.shipped_numerator == pytest.approx(n1, rel=1e-12)
        assert result.shipped_denominator == pytest.approx(d1, rel=1e-12)
        assert result.baseline_ratio == pytest.approx(n0 / d0, rel=1e-12)
        assert result.shipped_ratio == pytest.approx(n1 / d1, rel=1e-12)
        assert type(result).model_validate_json(result.model_dump_json()) == result
        assert pickle.loads(pickle.dumps(result)) == result
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_sum_matches_hand_computation(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        evidence = _moment_source(analysis).sitewide_evidence(
            analysis.metrics[0], include_ratio=False, population="triggered"
        )
        result = analysis.sitewide("revenue", population="triggered")
        arms = {arm.group_id: arm for arm in evidence.arm_stats}
        control = arms["C"]
        treatment = arms["T"]
        delta = treatment.mean_y() - control.mean_y()
        delta_se = math.sqrt(treatment.var_y() / treatment.n + control.var_y() / control.n)
        scaled_units = sum(evidence.population_counts.values())
        baseline = evidence.site_total - delta * treatment.n
        absolute = delta * scaled_units
        relative = absolute / baseline
        z = norm.ppf(0.975)
        absolute_se = delta_se * scaled_units
        relative_se = abs(scaled_units * evidence.site_total / baseline**2) * delta_se
        assert result.absolute_impact == pytest.approx(absolute, rel=1e-12)
        assert result.baseline_volume == pytest.approx(baseline, rel=1e-12)
        assert result.relative_impact == pytest.approx(relative, rel=1e-12)
        assert result.absolute_impact_se == pytest.approx(absolute_se, rel=1e-12)
        assert result.relative_impact_se == pytest.approx(relative_se, rel=1e-12)
        assert result.absolute_impact_lb == pytest.approx(absolute - z * absolute_se, rel=1e-12)
        assert result.absolute_impact_ub == pytest.approx(absolute + z * absolute_se, rel=1e-12)
        assert result.relative_impact_lb == pytest.approx(relative - z * relative_se, rel=1e-12)
        assert result.relative_impact_ub == pytest.approx(relative + z * relative_se, rel=1e-12)
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_assigned_and_triggered_share_provenance(tmp_path):
    analysis = _analysis(tmp_path)
    try:
        assigned = analysis.sitewide("revenue")
        triggered = analysis.sitewide("revenue", population="triggered")
        assert assigned.window == triggered.window
        assert assigned.snapshot_id == triggered.snapshot_id
        assert assigned.membership_cutoff is None
        assert triggered.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert assigned.assumptions == ("enrolled_units_exhaust_exposure",)
        assert assigned.allocation_balance is None
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_failed_balance_is_qualified_not_refused(tmp_path):
    rows = _events(n_per_arm=100, trigger_rate=1.0, effect=0.5).to_pylist()
    filtered = [
        row
        for row in rows
        if row["event"] != "saw_surface"
        or row["group_id"] == "C"
        or int(row["user_id"][1:]) % 20 == 0
    ]
    analysis = _analysis(tmp_path, pa.Table.from_pylist(filtered))
    try:
        result = analysis.sitewide("revenue", population="triggered")
        assert result.allocation_balance is not None
        assert result.allocation_balance.status == "failed"
    finally:
        analysis.close()


def _clustered_analysis(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definitions = _defs_yaml(end="2024-01-10").replace("    window_days: 7\n", "")
    definitions = definitions.replace(
        "    control_group: C\n",
        "    control_group: C\n    cluster: store_id\n",
    )
    definitions_path = tmp_path / "clustered-sitewide.yml"
    definitions_path.write_text(definitions)
    events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    events = events.append_column(
        "store_id",
        pa.array(
            [
                f"{str(user_id)[0]}store{int(str(user_id)[1:]) // 10}"
                for user_id in events["user_id"]
            ]
        ),
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=events)
    return Analysis.from_definitions(
        "exp",
        definitions_path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 17, tzinfo=dt.UTC)},
        ),
    )


@pytest.mark.filterwarnings("ignore:metric 'revenue':increment.errors.IncrementRuntimeWarning")
@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster")
@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_clustered_assigned_sitewide_artifact_matches_native(tmp_path):
    analysis = _clustered_analysis(tmp_path)
    adopted = None
    try:
        native = analysis.sitewide("revenue")
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts", "cluster_identity"),
        )
        artifact = adopted.sitewide("revenue")
        assert artifact.n_clusters == native.n_clusters
        assert artifact.absolute_impact == pytest.approx(native.absolute_impact, rel=1e-12)
        assert artifact.relative_impact == pytest.approx(native.relative_impact, rel=1e-12)
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


def test_windowed_assigned_sitewide_refuses_before_evidence_load(tmp_path):
    analysis = _analysis(
        tmp_path,
        _events(n_per_arm=100, trigger_rate=0.2, effect=0.5),
        metric_shape="windowed",
        observation_end="2024-01-16T00:00:00",
    )
    try:
        with pytest.raises(CapabilityError) as refused:
            analysis.sitewide("revenue")
        assert refused.value.code == "facade.analysis.sitewide_metric_unaligned"
    finally:
        analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_count_metric_sitewide_counts_events_alike_on_both_ingresses(tmp_path):
    rows = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5).to_pylist()
    conversions = [
        {**row, "event": "converted", "ts": dt.datetime(2024, 1, day), "value": 1.0}
        for row in rows
        if row["event"] == "enrolled" and int(row["user_id"][1:]) % 3 == 0
        for day in (3, 5)
    ]
    events = pa.concat_tables([pa.Table.from_pylist(rows), pa.Table.from_pylist(conversions)])
    analysis = _conversion_analysis(tmp_path, events, count_metric=True)
    adopted = None
    try:
        populations = ("assigned", "triggered")
        native = [analysis.sitewide("aaa_count", population=item) for item in populations]
        assert isinstance(native[0], SitewideImpact)
        assert native[0].site_total_volume == pytest.approx(len(conversions))
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)),
            analysis,
            kinds=("site_volume", "trigger_population", "assignment_counts"),
        )
        for population, expected in zip(populations, native, strict=True):
            artifact = adopted.sitewide("aaa_count", population=population)
            assert isinstance(expected, SitewideImpact)
            assert isinstance(artifact, SitewideImpact)
            assert artifact.site_total_volume == pytest.approx(expected.site_total_volume)
            assert artifact.absolute_impact == pytest.approx(expected.absolute_impact)
            assert artifact.relative_impact == pytest.approx(expected.relative_impact)
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)
