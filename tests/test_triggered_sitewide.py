from __future__ import annotations

import datetime as dt

import ibis
import pyarrow as pa
import pytest

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import CapabilityError
from increment.query.artifact_contract import ArtifactContractError
from tests.analysis_factory import _native_source, native_connection
from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
from tests.test_analysis_trigger import _defs_yaml, _events


def _analysis(tmp_path, events=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    definitions = tmp_path / "triggered-sitewide.yml"
    definitions.write_text(_defs_yaml(end="2024-01-10").replace("    window_days: 7\n", ""))
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=_events(n_per_arm=100, trigger_rate=0.2, effect=0.5) if events is None else events,
    )
    return Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
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
def test_triggered_sitewide_evidence_uses_calendar_window_member_outcomes(tmp_path):
    base_events = _events(n_per_arm=100, trigger_rate=0.2, effect=0.5)
    member_only_events = _add_late_trigger_and_outcome(base_events, include_outcome=False)
    with_late_outcome_events = _add_late_trigger_and_outcome(base_events, include_outcome=True)
    member_only = _analysis(tmp_path / "member-only", member_only_events)
    with_late_outcome = _analysis(tmp_path / "late-outcome", with_late_outcome_events)
    try:
        source = with_late_outcome._src
        evidence = source.sitewide_evidence(
            with_late_outcome.metrics[0], include_ratio=False, population="triggered"
        )
        reference = member_only._src.sitewide_evidence(
            member_only.metrics[0], include_ratio=False, population="triggered"
        )
        assert evidence.analysis_population == "triggered"
        assert evidence.window == (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
        assert evidence.membership_cutoff == dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
        assert evidence.population_counts == source.assignment_counts(population="triggered")
        assert evidence.assigned_counts == {"C": 100, "T": 100}
        assert evidence.population_counts["C"] == 17
        assert evidence.site_total == pytest.approx(
            with_late_outcome._src.sitewide_evidence(with_late_outcome.metrics[0]).site_total
        )
        assert {arm.group_id: arm.n for arm in evidence.arm_stats} == {
            arm.group_id: arm.n for arm in reference.arm_stats
        }
        assert {arm.group_id: arm.mean_y() for arm in evidence.arm_stats} == pytest.approx(
            {arm.group_id: arm.mean_y() for arm in reference.arm_stats}
        )
    finally:
        member_only.close()
        with_late_outcome.close()


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
        artifact = adopted._src.sitewide_evidence(
            adopted.metrics[0], include_ratio=False, population="triggered"
        )
        assert artifact.population_counts == native.population_counts
        assert artifact.assigned_counts == native.assigned_counts
        assert artifact.site_total == pytest.approx(native.site_total, rel=1e-12)
        assert {arm.group_id: arm.n for arm in artifact.arm_stats} == {
            arm.group_id: arm.n for arm in native.arm_stats
        }
        assert {arm.group_id: arm.mean_y() for arm in artifact.arm_stats} == pytest.approx(
            {arm.group_id: arm.mean_y() for arm in native.arm_stats}, rel=1e-12
        )
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
def test_triggered_sitewide_artifact_without_membership_refuses_with_republish_route(tmp_path):
    analysis = _analysis(tmp_path)
    adopted = None
    try:
        adopted = _publish_and_adopt(
            native_connection(_native_source(analysis)), analysis, kinds=("site_volume",)
        )
        with pytest.raises(ArtifactContractError) as raised:
            adopted._src.sitewide_evidence(
                adopted.metrics[0], include_ratio=False, population="triggered"
            )
        assert raised.value.code == "artifact.extension.missing"
        assert "republish" in str(raised.value.context["route"])
    finally:
        if adopted is None:
            analysis.close()
        else:
            _close_parity_analysis(adopted)


def test_triggered_sitewide_windowed_metric_refuses_before_data_load(tmp_path, monkeypatch):
    definitions = tmp_path / "windowed.yml"
    definitions.write_text(_defs_yaml(end="2024-01-10"))
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20))
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
        ),
    )
    try:
        monkeypatch.setattr(
            analysis._src,
            "sitewide_evidence",
            lambda *args, **kwargs: pytest.fail("loaded incompatible sitewide evidence"),
        )
        with pytest.raises(CapabilityError) as refused:
            analysis.sitewide("revenue", population="triggered")
        assert refused.value.code == "facade.analysis.sitewide_triggered_metric_unaligned"
        assert "calendar-aligned" in str(refused.value)
    finally:
        analysis.close()
