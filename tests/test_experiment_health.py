"""Behavioral contracts for callable experiment-health diagnostics."""

from collections.abc import Mapping, MutableMapping
from typing import Any, cast

import pyarrow as pa
import pytest
from pydantic import ValidationError

from increment import Analysis
from increment.results import ExperimentHealth, HealthCheck, SRMResult
from increment.semantics.design import Randomized


def test_experiment_health_repr_shows_source_and_check_count_without_storage_details():
    report = ExperimentHealth(
        checks=(),
        source_snapshot_id="snapshot-1",
        source_cutoff="2026-10-22",
    )
    display = repr(report)
    assert "\n" in display and "ExperimentHealth(" in display
    assert "checks=0" in display and "snapshot-1" in display
    assert not any(token in display for token in ("0x", "sha256:", "mappingproxy", "_v1"))


def test_health_models_are_frozen_and_own_nested_context() -> None:
    source = {"risk": ["counterfactual trigger invariance is unknown"]}
    check = HealthCheck(
        check="trigger_assumptions",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context=source,
    )
    source["risk"].append("changed after construction")
    report = ExperimentHealth(checks=(check,))
    assert report.checks[0].context["risk"] == ("counterfactual trigger invariance is unknown",)
    with pytest.raises((TypeError, AttributeError)):
        cast(MutableMapping[str, object], report.checks[0].context).update({"risk": ()})
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        type(report).__setattr__(report, "checks", ())


def test_health_json_replay_preserves_checks_and_risks() -> None:
    report = ExperimentHealth(
        checks=(
            HealthCheck(
                check="trigger_assumptions",
                analysis_population="triggered",
                availability="available",
                conclusion="descriptive",
                context={
                    "family_guarantee": None,
                    "risk": ("balanced rates do not establish trigger invariance",),
                },
            ),
        )
    )
    restored = ExperimentHealth.model_validate_json(report.model_dump_json())
    assert restored.checks == report.checks
    assert "passed" not in restored.model_dump()
    assert "healthy" not in restored.model_dump()
    assert "causal_validity" not in restored.model_dump()


def test_untriggered_summary_has_explicit_unavailable_triggered_rows() -> None:
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(
            control_group="C",
            allocation={"C": 0.5, "T": 0.5},
            allocation_scheme="independent",
        ),
    )
    report = analysis.health_check()
    assert [check.check for check in report.checks] == [
        "assigned_allocation",
        "triggered_balance",
        "trigger_rates",
        "trigger_maturity",
        "feed_completeness",
        "trigger_assumptions",
    ]
    assert report.checks[0].availability == "available"
    assert all(check.availability == "not_applicable" for check in report.checks[1:])
    assert all(check.conclusion is None for check in report.checks[1:])


@pytest.mark.integration
def test_balanced_trigger_rates_do_not_clear_assumption_risk(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    events = _events(n_per_arm=100, trigger_rate=0.5)
    event_rows = events.to_pylist()
    expected_rates = {}
    for arm in ("C", "T"):
        assigned_units = {
            row["user_id"]
            for row in event_rows
            if row["group_id"] == arm and row["event"] == "enrolled"
        }
        triggered_units = {
            row["user_id"]
            for row in event_rows
            if row["group_id"] == arm and row["event"] == "saw_surface"
        }
        expected_rates[arm] = len(triggered_units) / len(assigned_units)

    analysis = _analysis(tmp_path, events)
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    rates = next(check for check in report.checks if check.check == "trigger_rates")
    assumptions = next(check for check in report.checks if check.check == "trigger_assumptions")
    assert rates.availability == "available"
    assert dict(rates.rates) == pytest.approx(expected_rates)
    assert assumptions.context["counterfactual_trigger_invariance"].startswith("assumed")
    assert assumptions.context["trigger_order_stationarity"].startswith("assumed")
    assert assumptions.context["family_guarantee"] is None


@pytest.mark.integration
def test_forwarding_provider_health_check_preserves_trigger_evidence(tmp_path):
    from dataclasses import replace

    from increment.evidence import EvidenceProvider, evidence_provider
    from tests.analysis_factory import held_source
    from tests.test_analysis_trigger import _analysis, _events

    events = _events(n_per_arm=100, trigger_rate=0.5)
    event_rows = events.to_pylist()
    expected_rates = {
        arm: sum(row["group_id"] == arm and row["event"] == "saw_surface" for row in event_rows)
        / sum(row["group_id"] == arm and row["event"] == "enrolled" for row in event_rows)
        for arm in ("C", "T")
    }
    analysis = _analysis(tmp_path, events)
    source = cast(Any, held_source(analysis))
    adapter = evidence_provider(source)

    class ForwardingProvider:
        evidence = adapter.evidence
        context = adapter.context

        def __getattr__(self, name):
            return getattr(adapter, name)

        def moments(self, request):
            return adapter.moments(request)

        def unit_rows(self, request):
            return adapter.unit_rows(request)

        def method_fit(self, request):
            return adapter.method_fit(request)

        def breakout_method_moments(self, request):
            return adapter.breakout_method_moments(request)

        def compliance(self, request):
            return adapter.compliance(request)

        def compliance_series(self, request):
            return adapter.compliance_series(request)

        def counts(self, request):
            return adapter.counts(request)

        def capture_sequential(self, request):
            return adapter.capture_sequential(request)

        def scope(self, request):
            return adapter.scope(request)

    provider = ForwardingProvider()
    assert isinstance(provider, EvidenceProvider)
    original_state = analysis._state
    analysis._state = replace(original_state, source=provider)
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis._state = original_state
        analysis.close()

    rates = next(check for check in report.checks if check.check == "trigger_rates")
    maturity = [check for check in report.checks if check.check == "trigger_maturity"]
    feeds = [check for check in report.checks if check.check == "feed_completeness"]
    assert rates.availability == "available"
    assert dict(rates.rates) == pytest.approx(expected_rates)
    assert maturity and all(check.availability == "available" for check in maturity)
    assert feeds and any(check.feed == "events" for check in feeds)
    assert report.source_cutoff == "2024-01-16T00:00:00+00:00"


@pytest.mark.integration
def test_health_missing_certificates_are_unknown(tmp_path) -> None:
    import ibis

    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path), con)
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    assert triggered.availability == "unavailable"
    assert triggered.conclusion is None
    assert triggered.code == "source.native.trigger_evidence_required"
    completeness = [check for check in report.checks if check.check == "feed_completeness"]
    assert completeness
    assert all(check.availability == "unavailable" for check in completeness)
    assert all(check.conclusion is None for check in completeness)
    assert all(check.context["watermark"] is None for check in completeness)


@pytest.mark.integration
def test_pinned_health_report_repeats_with_identical_source_evidence(tmp_path) -> None:
    import ibis

    from increment._source_operations import DashboardSnapshotPayload
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path), con)
    reports = []

    def inspect_pinned(pinned: Analysis) -> DashboardSnapshotPayload:
        reports.extend(
            (
                pinned.health_check(expected_allocation={"C": 0.5, "T": 0.5}),
                pinned.health_check(expected_allocation={"C": 0.5, "T": 0.5}),
            )
        )
        return DashboardSnapshotPayload(
            allocation=None,
            allocation_refusal=None,
            allocation_history=None,
            allocation_history_refusal=None,
            estimates=(),
            group_data=(),
            explore=(),
        )

    try:
        analysis.dashboard_snapshot(inspect_pinned, metrics=analysis.metrics)
    finally:
        analysis.close()
        con.disconnect()

    assert len(reports) == 2
    assert reports[0] == reports[1]
    assert reports[0].source_snapshot_id == reports[1].source_snapshot_id


@pytest.mark.integration
def test_health_preserves_assigned_failure_beside_triggered_balance(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    rows = _events(n_per_arm=400, trigger_rate=0.5).to_pylist()
    triggered_c = {
        row["user_id"] for row in rows if row["event"] == "saw_surface" and row["group_id"] == "C"
    }
    remove = set(sorted(f"C{i}" for i in range(400) if f"C{i}" not in triggered_c)[:170])
    uneven = pa.Table.from_pylist([row for row in rows if row["user_id"] not in remove])
    analysis = _analysis(tmp_path, uneven, allocation_scheme="independent")
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    assert assigned.availability == "available"
    assert assigned.conclusion == "warning"
    assert assigned.srm is not None and assigned.srm.is_srm
    assert triggered.availability == "available"
    assert triggered.srm is not None and not triggered.srm.is_srm


@pytest.mark.integration
def test_health_unsupported_assignment_law_is_not_a_unit_arm_pass(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.5), allocation_scheme="blocked"
    )
    try:
        assigned = next(
            check
            for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
            if check.check == "assigned_allocation"
        )
    finally:
        analysis.close()
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.allocation_scheme_unsupported"
    assert assigned.conclusion is None
    assert assigned.srm is None


@pytest.mark.integration
def test_empty_native_assignment_returns_a_complete_health_report(tmp_path) -> None:
    import ibis

    from tests.test_analysis_trigger import _write_defs

    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE events (user_id VARCHAR, group_id VARCHAR, ts TIMESTAMP, "
        "event VARCHAR, value DOUBLE, experiment_id VARCHAR)"
    )
    analysis = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
    )
    try:
        report = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
        con.disconnect()

    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    assert assigned.availability == "unavailable"
    assert assigned.code == "estimation.diagnostics.total_observed_count"
    assert any(check.check == "triggered_balance" for check in report.checks)
    assert any(check.check == "trigger_assumptions" for check in report.checks)


def _health_snapshot_scenario(tmp_path, monkeypatch) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.analysis_factory import _native_source
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    original = _events(n_per_arm=100, trigger_rate=0.5)
    con.create_table("events", obj=original)
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    source = _native_source(analysis)
    pin = source._pinned_source
    inserted = False

    def pin_then_append(**kwargs):
        nonlocal inserted
        pinned = pin(**kwargs)
        if not inserted:
            inserted = True
            con.insert(
                "events",
                obj=pa.Table.from_pylist(
                    [
                        {
                            "user_id": f"{arm}new",
                            "group_id": arm,
                            "ts": dt.datetime(2024, 1, 2),
                            "event": event,
                            "value": 1.0,
                            "experiment_id": "exp",
                        }
                        for arm in ("C", "T")
                        for event in ("enrolled", "saw_surface", "revenue")
                    ]
                ),
            )
        return pinned

    monkeypatch.setattr(source, "_pinned_source", pin_then_append)
    try:
        first = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
        second = analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5})
    finally:
        analysis.close()
    first_assigned = next(check for check in first.checks if check.check == "assigned_allocation")
    second_assigned = next(check for check in second.checks if check.check == "assigned_allocation")
    first_rates = next(check for check in first.checks if check.check == "trigger_rates")
    second_rates = next(check for check in second.checks if check.check == "trigger_rates")
    first_srm, second_srm = first_assigned.srm, second_assigned.srm
    first_rate_values, second_rate_values = first_rates.rates, second_rates.rates
    assert first_srm is not None and second_srm is not None
    assert first_rate_values is not None and second_rate_values is not None
    assert first_srm.observed["C"] == 100
    assert second_srm.observed["C"] == 101
    assert first_rate_values["C"] == pytest.approx(
        sum(
            row["event"] == "saw_surface" and row["group_id"] == "C" for row in original.to_pylist()
        )
        / 100
    )
    assert second_rate_values["C"] == pytest.approx((first_rate_values["C"] * 100 + 1) / 101)
    assert first.source_snapshot_id and second.source_snapshot_id
    assert first.source_snapshot_id != second.source_snapshot_id
    assert first.source_cutoff == second.source_cutoff == cutoff.isoformat()
    restored = ExperimentHealth.model_validate_json(first.model_dump_json())
    assert restored.source_snapshot_id == first.source_snapshot_id
    assert restored.source_cutoff == first.source_cutoff


@pytest.mark.integration
def test_health_checks_share_one_snapshot(tmp_path, monkeypatch) -> None:
    _health_snapshot_scenario(tmp_path, monkeypatch)


@pytest.mark.integration
def test_health_second_call_sees_new_rows(tmp_path, monkeypatch) -> None:
    _health_snapshot_scenario(tmp_path, monkeypatch)


@pytest.mark.slow
@pytest.mark.integration
def test_health_artifact_trigger_extensions_are_optional_but_explicit(tmp_path) -> None:
    """Matched native/artifact diagnostics; portable shapes do not retain trigger evidence."""
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=30, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    native_report = native.health_check()
    artifact = _publish_and_adopt(
        con,
        native,
        kinds=("trigger_population", "assignment_counts", "trigger_measure_stats"),
    )
    try:
        report = artifact.health_check()
    finally:
        _close_parity_analysis(artifact)
    native_by_check = {check.check: check for check in native_report.checks}
    artifact_by_check = {check.check: check for check in report.checks}
    for name in (
        "assigned_allocation",
        "triggered_balance",
        "trigger_rates",
        "trigger_assumptions",
    ):
        assert artifact_by_check[name].model_dump(mode="json") == native_by_check[name].model_dump(
            mode="json"
        )
    native_maturity = {
        check.metric: check.model_dump(mode="json")
        for check in native_report.checks
        if check.check == "trigger_maturity"
    }
    artifact_maturity = {
        check.metric: check.model_dump(mode="json")
        for check in report.checks
        if check.check == "trigger_maturity"
    }
    assert artifact_maturity == native_maturity
    native_completeness = {
        (
            check.availability,
            check.conclusion,
            check.code,
            check.context.get("watermark"),
            check.context.get("reason"),
        )
        for check in native_report.checks
        if check.check == "feed_completeness"
    }
    artifact_completeness = [check for check in report.checks if check.check == "feed_completeness"]
    assert {
        (
            check.availability,
            check.conclusion,
            check.code,
            check.context.get("watermark"),
            check.context.get("reason"),
        )
        for check in artifact_completeness
    } == native_completeness
    assert any(check.feed == "trigger_population" for check in artifact_completeness)
    assert any(check.feed == "trigger_measure_stats:revenue" for check in artifact_completeness)


def test_health_models_round_trip_pickle_deepcopy_and_nested_immutability() -> None:
    import copy
    import pickle

    unavailable = HealthCheck(
        check="triggered_balance",
        analysis_population="triggered",
        availability="unavailable",
        code="source.native.trigger_evidence_required",
        context={"reason": "no trigger counts", "details": {"feeds": ["events"]}},
    )
    rates = HealthCheck(
        check="trigger_rates",
        analysis_population="triggered",
        availability="available",
        conclusion="descriptive",
        context={"description": "observed trigger shares"},
        rates={"C": 0.25, "T": 0.25},
    )
    report = ExperimentHealth(checks=(unavailable, rates))
    for restored in (
        pickle.loads(pickle.dumps(report)),
        copy.deepcopy(report),
        ExperimentHealth.model_validate_json(report.model_dump_json()),
    ):
        assert restored == report
        details = cast(Mapping[str, object], restored.checks[0].context["details"])
        assert cast(tuple[str, ...], details["feeds"]) == ("events",)
        assert restored.checks[1].rates == {"C": 0.25, "T": 0.25}
        with pytest.raises((TypeError, AttributeError)):
            cast(MutableMapping[str, object], restored.checks[0].context["details"]).update(
                {"feeds": ()}
            )

        with pytest.raises((TypeError, AttributeError)):
            cast(MutableMapping[str, float], restored.checks[1].rates).update({"C": 0.5})


@pytest.mark.integration
def test_health_artifact_without_trigger_extensions_is_unavailable(tmp_path) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path, allocation_scheme="independent"),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    artifact = _publish_and_adopt(con, native, kinds=("assignment_counts",))
    try:
        report = artifact.health_check()
    finally:
        _close_parity_analysis(artifact)
    for name in ("triggered_balance", "trigger_rates"):
        check = next(row for row in report.checks if row.check == name)
        assert check.availability == "unavailable"
        assert check.conclusion is None
        assert check.code is not None


@pytest.mark.slow
@pytest.mark.integration
def test_triggered_health_without_declared_allocation_is_unavailable_for_native_and_artifact(
    tmp_path,
) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.parity_harness.cases import _close_parity_analysis, _publish_and_adopt
    from tests.test_analysis_trigger import _events, _write_defs

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    native = Analysis.from_definitions(
        "exp",
        _write_defs(tmp_path),
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    artifact = _publish_and_adopt(
        con,
        native,
        kinds=("trigger_population", "assignment_counts", "trigger_measure_stats"),
    )
    try:
        reports = (native.health_check(), artifact.health_check())
    finally:
        _close_parity_analysis(artifact)
        native.close()
    for report in reports:
        check = next(row for row in report.checks if row.check == "triggered_balance")
        assert check.availability == "unavailable"
        assert check.conclusion is None
        assert check.code == "integrity.allocation_scheme_missing"
        assert "allocation=" in str(check.context["reason"])


@pytest.mark.integration
def test_native_health_completeness_omits_unused_catalog_feed(tmp_path) -> None:
    import ibis

    from tests.test_analysis_trigger import _defs_yaml, _events

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    definitions = tmp_path / "defs.yml"
    definitions.write_text(
        _defs_yaml().replace(
            "fact_sources:\n",
            "fact_sources:\n"
            "  - name: unused\n"
            "    sql: SELECT * FROM unused\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts:\n"
            "      - name: unused_fact\n"
            "        column: value\n",
        )
    )
    analysis = Analysis.from_definitions("exp", definitions, con)
    try:
        report = analysis.health_check()
    finally:
        analysis.close()
    feeds = {check.feed for check in report.checks if check.check == "feed_completeness"}
    assert "unused" not in feeds
    assert "events" in feeds


@pytest.mark.integration
def test_native_health_includes_unwatermarked_separate_uptake_feed(tmp_path) -> None:
    import ibis

    from tests.test_analysis_trigger import _defs_yaml, _events

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    definitions = tmp_path / "defs.yml"
    definitions.write_text(
        _defs_yaml()
        .replace(
            "fact_sources:\n",
            "fact_sources:\n"
            "  - name: uptake_events\n"
            "    sql: SELECT * FROM events\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts:\n"
            "      - name: clicked\n"
            "        column: null\n",
        )
        .replace(
            "    control_group: C\n",
            "    control_group: C\n"
            "    design:\n"
            "      mechanism: encouragement\n"
            "      uptake: {fact: clicked, window_days: 1}\n"
            "      one_sided: true\n"
            "      exclusion_restriction: {justification: gates outcome}\n",
        )
    )
    analysis = Analysis.from_definitions("exp", definitions, con)
    try:
        report = analysis.health_check()
    finally:
        analysis.close()
    uptake = next(
        check
        for check in report.checks
        if check.check == "feed_completeness" and check.feed == "uptake_events"
    )
    assert uptake.availability == "unavailable"
    assert uptake.context["watermark"] is None


@pytest.mark.integration
def test_sql_trigger_completeness_is_reported_unavailable(tmp_path) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.test_analysis_trigger import _defs_yaml, _events

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    definitions = tmp_path / "defs.yml"
    definitions.write_text(
        _defs_yaml()
        .replace(
            "  - name: saw_surface\n    fact: saw_surface\n",
            "  - name: events\n"
            "    sql: SELECT user_id, ts, group_id FROM events WHERE event = 'saw_surface'\n",
        )
        .replace("trigger: saw_surface", "trigger: events")
    )
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    try:
        report = analysis.health_check()
    finally:
        analysis.close()
    checks = [
        check
        for check in report.checks
        if check.check == "feed_completeness" and check.feed == "events"
    ]
    assert len(checks) == 2
    fact_feed = next(check for check in checks if check.context["evidence_kind"] == "native_feed")
    sql_trigger = next(
        check for check in checks if check.context["evidence_kind"] == "native_sql_trigger"
    )
    assert fact_feed.availability == "available"
    assert sql_trigger.availability == "unavailable"
    assert sql_trigger.context["watermark"] is None
    assert "SQL-backed" in str(sql_trigger.context["reason"])


@pytest.mark.integration
def test_sql_assignment_completeness_does_not_inherit_same_named_feed_watermark(
    tmp_path,
) -> None:
    import datetime as dt

    import ibis

    from increment import SourceSnapshotEvidence
    from tests.test_analysis_trigger import _defs_yaml, _events

    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=20, trigger_rate=0.5))
    cutoff = dt.datetime(2024, 1, 16, tzinfo=dt.UTC)
    definitions = tmp_path / "defs.yml"
    definitions.write_text(
        _defs_yaml()
        .replace(
            "  - name: assignment\n    fact: enrolled\n",
            "  - name: events\n"
            "    sql: SELECT user_id AS unit_id, ts, group_id FROM events WHERE event = 'enrolled'\n",
        )
        .replace("exposure: assignment", "exposure: events")
    )
    analysis = Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=cutoff,
            complete_through_by_feed={"events": cutoff},
        ),
    )
    try:
        report = analysis.health_check()
    finally:
        analysis.close()

    checks = [
        check
        for check in report.checks
        if check.check == "feed_completeness" and check.feed == "events"
    ]
    assert len(checks) == 2
    fact_feed = next(check for check in checks if check.context["evidence_kind"] == "native_feed")
    sql_assignment = next(
        check for check in checks if check.context["evidence_kind"] == "native_sql_assignment"
    )
    assert fact_feed.availability == "available"
    assert sql_assignment.availability == "unavailable"
    assert sql_assignment.context["watermark"] is None
    assert "SQL-backed assignment" in str(sql_assignment.context["reason"])


def test_health_validation_errors_are_coded() -> None:
    from increment.errors import InvalidRequestError

    available = HealthCheck(
        check="assigned_allocation",
        analysis_population="assigned",
        availability="available",
        conclusion="not_rejected",
        context={},
    )
    unavailable = HealthCheck(
        check="triggered_balance",
        analysis_population="triggered",
        availability="unavailable",
        context={"reason": "unknown"},
    )
    with pytest.raises(InvalidRequestError) as raised:
        ExperimentHealth(checks=(unavailable, available))
    assert raised.value.code == "estimation.health.experiment.stable_order"


def test_health_unit_panel_reports_only_supported_population() -> None:
    panel = Analysis.from_unit_panel(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "day": ["2025-01-01"] * 4,
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        metrics={"value": "mean"},
        control="C",
    )
    report = panel.health_check()
    assert report.checks[0].availability == "unavailable"
    assert report.checks[0].code == "integrity.allocation_scheme_missing"


def test_run_leaves_undeclared_srm_visible_in_health_check() -> None:
    import warnings

    from tests.warning_codes import warning_codes

    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        control="C",
        metrics={"value": "mean"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis.run()
    assert "integrity.srm_not_checked" not in warning_codes(caught)

    assigned = next(
        check for check in analysis.health_check().checks if check.check == "assigned_allocation"
    )
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.allocation_scheme_missing"
    assert assigned.context["route"]
    # The suggested fixed look works once target shares are supplied.
    fixed = analysis.srm(inference="fixed", expected={"C": 0.5, "T": 0.5})
    assert isinstance(fixed, SRMResult) and fixed.is_srm is False


def test_health_switchback_does_not_substitute_unit_arm_srm() -> None:
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    switchback = Analysis.from_switchback_panel(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "cycle": cycle,
                    "period": period,
                    "step": step,
                    "group": group,
                    "value": float(cycle + period + step),
                }
                for unit, order in (("u1", ("C", "T")), ("u2", ("T", "C")))
                for cycle in range(2)
                for period, group in enumerate(order)
                for step in range(2)
            ]
        ),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(control_group="C", allocation={"C": 0.5, "T": 0.5}),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    assigned = switchback.health_check().checks[0]
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.switchback_assignment_law"
    assert assigned.srm is None


def test_health_moments_replay_does_not_reconstruct_trigger_population() -> None:
    from tests.test_analysis_moments import _legacy_fixed10_rows

    replay = Analysis.from_moments(
        _legacy_fixed10_rows(trigger_name="saw_surface"),
        metrics={"revenue": "mean"},
        control="control",
    )
    report = replay.health_check()
    triggered_balance = next(check for check in report.checks if check.check == "triggered_balance")
    trigger_rates = next(check for check in report.checks if check.check == "trigger_rates")
    completeness = next(check for check in report.checks if check.check == "feed_completeness")
    assert triggered_balance.availability == "unavailable"
    assert triggered_balance.code is not None
    assert trigger_rates.availability == "unavailable"
    assert trigger_rates.code is not None
    assert completeness.availability == "unavailable"
    assert completeness.context["watermark"] is None


def test_health_unavailable_is_not_nonrejection() -> None:
    check = HealthCheck(
        check="triggered_balance",
        analysis_population="triggered",
        availability="unavailable",
        code="source.native.trigger_evidence_required",
        context={"reason": "trigger counts were not retained"},
    )
    assert check.conclusion is None
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        HealthCheck(
            check="triggered_balance",
            analysis_population="triggered",
            availability="unavailable",
            conclusion="not_rejected",
            code="source.native.trigger_evidence_required",
            context={"reason": "trigger counts were not retained"},
        )
    assert raised.value.code == "estimation.health.check.unavailable_conclusion"


def test_health_expected_allocation_completes_independent_design_admission() -> None:
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(control_group="C", allocation_scheme="independent"),
    )
    assigned = next(
        check
        for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
        if check.check == "assigned_allocation"
    )
    assert assigned.availability == "available"
    assert assigned.srm is not None
    assert assigned.srm.expected == {"C": 0.5, "T": 0.5}


def test_health_extreme_valid_srm_statistic_round_trips_through_json() -> None:
    import math

    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(control_group="C", allocation_scheme="independent"),
    )
    report = analysis.health_check(expected_allocation={"C": 5e-324, "T": 1.0})
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    assert assigned.srm is not None
    assert math.isinf(assigned.srm.chi2_stat)
    assert report.source_snapshot_id is not None

    restored = ExperimentHealth.model_validate_json(report.model_dump_json())
    assert restored.checks == report.checks
    assert restored.source_snapshot_id == report.source_snapshot_id


@pytest.mark.integration
def test_health_empty_trigger_population_is_unavailable_not_a_report_failure(tmp_path) -> None:
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.0), allocation_scheme="independent"
    )
    try:
        report = analysis.health_check()
    finally:
        analysis.close()
    assigned = next(check for check in report.checks if check.check == "assigned_allocation")
    triggered = next(check for check in report.checks if check.check == "triggered_balance")
    rates = next(check for check in report.checks if check.check == "trigger_rates")
    assert assigned.availability == "available"
    assert triggered.availability == "unavailable"
    assert triggered.code == "estimation.diagnostics.total_observed_count"
    assert triggered.conclusion is None
    assert rates.availability == "available"
    assert rates.rates == {"C": 0.0, "T": 0.0}


@pytest.mark.integration
def test_health_validates_expected_allocation_before_empty_trigger_counts(tmp_path) -> None:
    from increment.errors import InvalidRequestError
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(
        tmp_path, _events(n_per_arm=20, trigger_rate=0.0), allocation_scheme="blocked"
    )
    try:
        with pytest.raises(InvalidRequestError) as caught:
            analysis.health_check(expected_allocation={"C": 0.0, "T": 1.0})
    finally:
        analysis.close()
    assert caught.value.code == "estimation.diagnostics.every_expected_share_positive"


def test_health_srm_report_supports_pickle_and_deepcopy() -> None:
    import copy
    import pickle

    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(
            control_group="C",
            allocation={"C": 0.5, "T": 0.5},
            allocation_scheme="independent",
        ),
    )
    report = analysis.health_check()
    assigned = report.checks[0]
    assert assigned.srm is not None
    assert assigned.availability == "available"
    for restored in (pickle.loads(pickle.dumps(report)), copy.deepcopy(report)):
        restored_srm = restored.checks[0].srm
        assert restored_srm is not None
        assert restored_srm == assigned.srm
        assert restored_srm.observed == assigned.srm.observed


@pytest.mark.integration
def test_health_assumption_discloses_triggered_family_guarantee(tmp_path) -> None:
    from tests.test_triggered_sequential import _FAMILY_PLAN, _analysis, _events

    secondaries = ("aux_a", "aux_b")
    analysis, _ = _analysis(
        tmp_path,
        _events(n_per_arm=20, secondaries=secondaries),
        plan=_FAMILY_PLAN,
        secondaries=secondaries,
    )
    try:
        assumptions = next(
            check
            for check in analysis.health_check().checks
            if check.check == "trigger_assumptions"
        )
    finally:
        analysis.close()
    assert assumptions.context["family_guarantee"] == "asymptotic_sequential"
    assert assumptions.context["family_guarantee_status"] == "assumed, not verified"
    assert assumptions.context["no_effect_before_trigger"] is not None
    assert assumptions.context["triggered_look_predictability"] is not None
    restored = type(assumptions).model_validate_json(assumptions.model_dump_json())
    assert restored.context == assumptions.context
    assert "triggered_look_policy" not in assumptions.context


def test_health_missing_assignment_scheme_suggests_supported_routes() -> None:
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "value": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"value": "mean"},
        design=Randomized(control_group="C"),
    )
    assigned = next(
        check
        for check in analysis.health_check(expected_allocation={"C": 0.5, "T": 0.5}).checks
        if check.check == "assigned_allocation"
    )
    assert assigned.availability == "unavailable"
    assert assigned.code == "integrity.allocation_scheme_missing"
    assert assigned.context["route"]
    fixed = analysis.srm(inference="fixed", expected={"C": 0.5, "T": 0.5})
    assert isinstance(fixed, SRMResult) and fixed.is_srm is False


def test_health_snapshot_fingerprint_canonicalizes_nonfinite_evidence() -> None:
    import hashlib
    from types import SimpleNamespace

    from increment._canonical import canonical_json_bytes
    from increment.sources import _health_snapshot_id

    source = SimpleNamespace(_source_identity_record={"artifact": "stable"})
    evidence = [{"rate": float("inf"), "values": [float("nan"), float("-inf")]}]
    expected = {
        "source": {"artifact": "stable"},
        "cutoff": "2024-01-16T00:00:00+00:00",
        "health_evidence": [
            {
                "rate": {"$nonfinite_float": "inf"},
                "values": [
                    {"$nonfinite_float": "nan"},
                    {"$nonfinite_float": "-inf"},
                ],
            }
        ],
    }

    assert (
        _health_snapshot_id(source, "2024-01-16T00:00:00+00:00", evidence)
        == "sha256:" + hashlib.sha256(canonical_json_bytes(expected)).hexdigest()
    )
