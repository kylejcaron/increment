from __future__ import annotations

from typing import Any, cast

import pytest

from increment.readouts import run
from tests.analysis_factory import held_source, lift_rows
from tests.source_conformance import definitions_source


@pytest.mark.integration
def test_run_captures_scope_and_persists_mode_metadata(tmp_path):
    analysis, _ = definitions_source(
        tmp_path, trigger=True, sql_exposures=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    try:
        rows = run(source, coherence="captured")
        assert rows
        metadata = rows.metadata
        assert metadata is not None
        assert metadata.coherence == "captured"
        assert set(metadata.effective_modes.values()) == {"captured"}
        assert "events" in metadata.effective_modes
        assert "revenue" not in metadata.effective_modes
        assert {"exposure:assignment", "exposure:trigger"} <= set(metadata.effective_modes)
    finally:
        analysis.close()


@pytest.mark.integration
def test_run_live_mode_is_retained_and_qualified(tmp_path):
    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    try:
        rows = run(source, coherence="live")
        assert rows
        metadata = rows.metadata
        assert metadata is not None
        assert metadata.coherence == "live"
        assert set(metadata.effective_modes) == {"events"}
        import json

        import narwhals as nw

        from increment.tables import estimates_to_readout, readout_table

        frame = nw.from_native(rows.to_frame(backend="polars"), eager_only=True)
        expected_modes = {"events": "live"}
        assert frame["coherence"].to_list() == ["live"] * len(rows)
        assert [json.loads(value) for value in frame["effective_modes"].to_list()] == [
            expected_modes
        ] * len(rows)
        exported = estimates_to_readout(rows)
        assert all(row["coherence"] == "live" for row in exported)
        assert all(json.loads(row["effective_modes"]) == expected_modes for row in exported)
        pytest.importorskip("coeftable")
        html = readout_table(rows).gt().as_raw_html()
        assert "Coherence: live" in html
    finally:
        analysis.close()


@pytest.mark.integration
def test_live_mode_includes_sql_exposure_streams(tmp_path):
    analysis, _ = definitions_source(
        tmp_path, trigger=True, sql_exposures=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    try:
        rows = run(source, coherence="live")
        assert rows.metadata is not None
        assert set(rows.metadata.effective_modes) == {
            "events",
            "exposure:assignment",
            "exposure:trigger",
        }
        assert set(rows.metadata.effective_modes.values()) == {"live"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_mutable_sql_panel_is_not_reported_as_captured():
    from increment import EvidenceScope
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    con = source._con
    con.create_table("mutable_summary", obj=con.to_pyarrow(source._summary))
    source._summary = con.table("mutable_summary")
    context = EvidenceExecutionContext(
        native_scope_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            snapshot=True,
        ),
        run_id="mutable-sql-summary",
    )
    try:
        context.open()
        con.raw_sql("UPDATE mutable_summary SET ref_y = ref_y + 100 WHERE group_id = 'treatment'")
        rows = run(source, execution_context=context)
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence == "live"
        assert set(rows.metadata.effective_modes.values()) == {"live"}
    finally:
        context.close()
        source.close()


@pytest.mark.integration
def test_native_scope_provider_yields_typed_pinned_provider():
    from increment import EvidenceScope
    from increment.evidence import MomentsRequest, request_evidence
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    metric = source.context.metrics[0]
    context = EvidenceExecutionContext(
        native_scope_provider(source),
        EvidenceScope(
            metrics=(metric,),
            configs=tuple(source.context.configs),
        ),
        run_id="typed-pinned-provider",
        coherence="live",
    )
    try:
        scoped = context.open()
        rows = request_evidence(scoped, MomentsRequest(metric))
        assert rows
    finally:
        context.close()
        source.close()


@pytest.mark.integration
def test_in_memory_moments_are_captured_without_snapshot_operation():
    from increment import EvidenceScope
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.source_conformance import moments_source

    source = moments_source()
    context = EvidenceExecutionContext(
        native_scope_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
        ),
        run_id="immutable-moments",
    )
    try:
        context.open()
        assert context.coherence == "captured"
        assert set(context.effective_modes.values()) == {"captured"}
    finally:
        context.close()
        source.close()


@pytest.mark.integration
def test_empty_selection_and_invalid_request_do_not_open_execution_scope(tmp_path):
    from increment.errors import CodedError

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))

    class ScopeGuard:
        evidence = source.evidence
        context = source.context

        def moments(self, request):
            return source.moments(request)

        def breakout_method_moments(self, request):
            return source.breakout_method_moments(request)

        def unit_rows(self, request):
            return source.unit_rows(request)

        def method_fit(self, request):
            return source.method_fit(request)

        def compliance(self, request):
            return source.compliance(request)

        def compliance_series(self, request):
            return source.compliance_series(request)

        def counts(self, request):
            from increment.evidence import CountEvidence

            return CountEvidence(
                request.population,
                "assignment",
                source.assignment_counts(population=request.population),
            )

        def capture_sequential(self, request):
            return source.capture_sequential(request)

        def scope(self, request):
            pytest.fail(f"validated request opened an evidence scope: {request!r}")

    guard = ScopeGuard()
    assert run(guard, metrics=[]) == []
    with pytest.raises(CodedError) as readout_error:
        run(guard, metrics=["not-a-declared-metric"])
    assert readout_error.value.code == "facade.analysis_config.unknown_metric_declared"

    analysis.close()
    assert analysis.run(metrics=[]) == []
    with pytest.raises(CodedError) as facade_error:
        analysis.run(metrics=["not-a-declared-metric"])
    assert facade_error.value.code == "facade.analysis_config.unknown_metric_declared"


@pytest.mark.integration
def test_capture_keeps_property_used_by_selected_metric_filter(tmp_path):
    import datetime as dt

    import ibis
    import yaml

    from increment import Analysis

    rows = []
    for unit, arm, eligible in (
        ("c1", "control", True),
        ("c2", "control", False),
        ("t1", "treatment", True),
        ("t2", "treatment", False),
    ):
        session = int(eligible)
        rows.extend(
            (
                {
                    "user_id": unit,
                    "ts": dt.datetime(2026, 1, 1, 9),
                    "event": "assignment",
                    "group_id": arm,
                    "sess": session,
                },
                {
                    "user_id": unit,
                    "ts": dt.datetime(2026, 1, 1, 12),
                    "event": "purchase",
                    "group_id": None,
                    "sess": session,
                },
            )
        )
    definitions = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "assignment", "column": None},
                    {"name": "purchase", "column": None},
                    {"name": "unused_sessions", "column": "sess"},
                ],
                "properties": [{"name": "sess", "column": "sess", "dtype": "int"}],
            }
        ],
        "exposures": [{"name": "assignment", "fact": "assignment"}],
        "metrics": [
            {
                "type": "conversion",
                "name": "purchase_rate",
                "entity": "user_id",
                "fact": "purchase",
                "window_days": 1,
                "filters": [{"property": "sess", "op": "gt", "values": [0]}],
            }
        ],
        "experiments": [
            {
                "name": "filter-capture",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "control_group": "control",
                "plan": {"primary": "purchase_rate"},
            }
        ],
    }
    path = tmp_path / "filter-capture.yaml"
    path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)
    analysis = Analysis.from_definitions("filter-capture", path, con)
    try:
        source = cast(Any, held_source(analysis))
        live = run(source, metrics=["purchase_rate"], coherence="live")
        captured = run(source, metrics=["purchase_rate"], coherence="captured")

        live_lifts = [row.lift for row in live]
        captured_lifts = [row.lift for row in captured]
        assert all(lift is not None for lift in live_lifts + captured_lifts)
        assert [lift.value for lift in captured_lifts if lift is not None] == [
            lift.value for lift in live_lifts if lift is not None
        ]
    finally:
        analysis.close()


@pytest.mark.integration
def test_compliance_only_triggered_run_validates_within_one_capture(tmp_path):
    import datetime as dt

    import ibis
    import yaml

    from increment import Analysis
    from increment.errors import CodedError

    rows = [
        {
            "user_id": "c1",
            "ts": dt.datetime(2026, 1, 1, 9),
            "event": "assignment",
            "group_id": "control",
            "revenue": None,
        },
        {
            "user_id": "c1",
            "ts": dt.datetime(2026, 1, 1, 10),
            "event": "trigger",
            "group_id": "control",
            "revenue": None,
        },
        {
            "user_id": "t1",
            "ts": dt.datetime(2026, 1, 1, 9),
            "event": "assignment",
            "group_id": "treatment",
            "revenue": None,
        },
        {
            "user_id": "c1",
            "ts": dt.datetime(2026, 1, 1, 12),
            "event": "purchase",
            "group_id": None,
            "revenue": 1.0,
        },
        {
            "user_id": "t1",
            "ts": dt.datetime(2026, 1, 1, 12),
            "event": "purchase",
            "group_id": None,
            "revenue": 2.0,
        },
        {
            "user_id": "c2",
            "ts": dt.datetime(2026, 1, 1, 9),
            "event": "assignment",
            "group_id": "control",
            "revenue": None,
        },
        {
            "user_id": "c2",
            "ts": dt.datetime(2026, 1, 1, 12),
            "event": "purchase",
            "group_id": None,
            "revenue": 1.5,
        },
        {
            "user_id": "t2",
            "ts": dt.datetime(2026, 1, 1, 9),
            "event": "assignment",
            "group_id": "treatment",
            "revenue": None,
        },
        {
            "user_id": "t2",
            "ts": dt.datetime(2026, 1, 1, 12),
            "event": "purchase",
            "group_id": None,
            "revenue": 2.5,
        },
    ]
    definitions = {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [
                    {"name": "assignment", "column": None},
                    {"name": "trigger", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ],
            }
        ],
        "exposures": [
            {"name": "assignment", "fact": "assignment"},
            {"name": "trigger", "fact": "trigger"},
        ],
        "metrics": [
            {
                "type": "mean",
                "name": "revenue",
                "entity": "user_id",
                "fact": "purchase",
                "aggregation": "sum",
                "window_days": 1,
            }
        ],
        "experiments": [
            {
                "name": "triggered-compliance",
                "exposure": "assignment",
                "trigger": "trigger",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "control_group": "control",
                "plan": {"primary": "revenue"},
                "design": {
                    "mechanism": "encouragement",
                    "uptake": {"fact": "purchase"},
                    "one_sided": False,
                },
            }
        ],
    }
    path = tmp_path / "triggered-compliance.yaml"
    path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)
    from increment import SourceSnapshotEvidence

    evidence_time = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    analysis = Analysis.from_definitions(
        "triggered-compliance",
        path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(evidence_time, {"events": evidence_time}),
    )
    source = cast(Any, held_source(analysis))
    original = source._pinned_source
    captures = 0

    def count_capture(*args, **kwargs):
        nonlocal captures
        pinned = original(*args, **kwargs)
        captures += pinned is not source
        return pinned

    source._pinned_source = count_capture
    try:
        with pytest.raises(CodedError) as error:
            analysis.run(metrics=[], estimands=["compliance"])

        assert error.value.code == "query.integrity.trigger_arm_missing"
        assert captures == 1
    finally:
        analysis.close()


@pytest.mark.integration
def test_whole_window_assigned_and_triggered_reads_share_capture(tmp_path, monkeypatch):
    from contextlib import contextmanager

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    source = cast(Any, held_source(analysis))
    original = source.readout_snapshot
    snapshots = 0

    @contextmanager
    def count_snapshot(*args, **kwargs):
        nonlocal snapshots
        snapshots += 1
        with original(*args, **kwargs) as pinned:
            source._con.raw_sql(
                "UPDATE raw_events SET revenue = revenue + 100 "
                "WHERE event = 'purchase' AND user_id = 'u3'"
            )
            yield pinned

    monkeypatch.setattr(source, "readout_snapshot", count_snapshot)
    try:
        rows = analysis.run()
        assert {row.analysis_population for row in rows} == {"assigned", "triggered"}
        assert snapshots == 1
        values = {row.analysis_population: row.require_lift().value for row in rows}
        assert values["assigned"] == pytest.approx(2.0)
        assert values["triggered"] == pytest.approx(2.0)
    finally:
        analysis.close()


@pytest.mark.integration
def test_borrowed_context_rejects_a_different_source_before_read(tmp_path):
    from increment import EvidenceScope
    from increment.errors import CodedError
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext

    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first, _ = definitions_source(tmp_path / "first", return_analysis=True)
    second, _ = definitions_source(tmp_path / "second", return_analysis=True)
    first_source = cast(Any, held_source(first))
    second_source = cast(Any, held_source(second))
    context = EvidenceExecutionContext(
        evidence_provider(first_source),
        EvidenceScope(
            metrics=tuple(first_source.context.metrics),
            configs=tuple(first_source.context.configs),
            execution=True,
        ),
        run_id="cross-source",
    )
    try:
        context.open()
        with pytest.raises(CodedError) as refused:
            run(second_source, execution_context=context)
        assert refused.value.code == "kernel.context.undeclared_stream"
    finally:
        context.close()
        first.close()
        second.close()


@pytest.mark.integration
def test_borrowed_context_refuses_an_undeclared_metric_configuration(tmp_path):
    from increment import EvidenceScope
    from increment.errors import CodedError
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            execution=True,
        ),
        run_id="undeclared-config",
    )
    try:
        context.open()
        with pytest.raises(CodedError) as refused:
            run(source, execution_context=context)
        assert refused.value.code == "kernel.context.undeclared_stream"
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_collection_concat_preserves_and_checks_coherence_metadata(tmp_path):
    from increment.errors import CodedError

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    try:
        captured = run(source, coherence="captured")
        captured_again = run(source, coherence="captured")
        live = run(source, coherence="live")
        assert captured.concat(captured_again).metadata.coherence == "captured"
        assert live.concat(run(source, coherence="live")).metadata.coherence == "live"
        with pytest.raises(CodedError) as refused:
            captured.concat(live)
        assert refused.value.code == "readout.collection.concat_metadata_mismatch"
    finally:
        analysis.close()


@pytest.mark.integration
def test_artifact_run_remains_captured_when_live_is_requested(tmp_path):
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    try:
        rows = run(cast(Any, held_source(analysis)), coherence="live")
        assert rows.metadata is not None
        assert rows.metadata.coherence == "captured"
        assert rows.metadata.effective_modes
        assert set(rows.metadata.effective_modes.values()) == {"captured"}
        manifest = cast(Any, held_source(analysis))._manifest
        identity = f"artifact:{manifest.artifact_id}:{manifest.generation_id}"
        assert rows.metadata.effective_modes == {identity: "captured"}
        from increment import EvidenceScope, evidence_provider
        from increment.kernel.context import EvidenceExecutionContext

        source = cast(Any, held_source(analysis))
        context = EvidenceExecutionContext(
            evidence_provider(source),
            EvidenceScope(
                metrics=tuple(source.context.metrics),
                configs=tuple(source.context.configs),
            ),
            run_id="artifact-borrow",
            coherence="live",
        )
        context.open()
        try:
            borrowed = run(source, coherence="live", execution_context=context)
            assert borrowed.metadata is not None
            assert borrowed.metadata.coherence == "captured"
        finally:
            context.close()
    finally:
        analysis.close()


@pytest.mark.integration
def test_compliance_only_read_uses_borrowed_context():
    from increment import EvidenceScope
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext
    from tests.test_compliance_summary import witness

    source = witness()
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            uptake_facts=("clicked",),
            execution=True,
        ),
        run_id="compliance-only",
    )
    try:
        context.open()
        rows = run(evidence_provider(source), estimands=("compliance",), execution_context=context)
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence == "captured"
        assert set(rows.metadata.effective_modes.values()) == {"captured"}
        assert "fact:clicked" in rows.metadata.effective_modes
    finally:
        context.close()


@pytest.mark.integration
def test_whole_window_captures_call_time_method_overrides(tmp_path):
    from increment import Method

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    try:
        rows = analysis.run(decision_method=Method(name="unadjusted"))
        assert rows
        assert {row.method for row in rows} == {"unadjusted"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_trigger_validation_uses_the_pinned_whole_window_source(tmp_path, monkeypatch):
    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    validated_sources = []
    acquired_sources = []
    original = analysis._validate_trigger_fires_in_every_arm
    from increment.readouts import _evidence

    original_load_counts = _evidence.load_counts

    def load_counts(source, request):
        acquired_sources.append((source, request))
        return original_load_counts(source, request)

    def validate(source=None):
        validated_sources.append(source)
        return original() if source is None else original(source)

    monkeypatch.setattr(_evidence, "load_counts", load_counts)
    monkeypatch.setattr(analysis, "_validate_trigger_fires_in_every_arm", validate)
    try:
        rows = analysis.run()
        assert rows
        assert len(validated_sources) == 1
        pinned = validated_sources[0]
        assert pinned is not None
        assert pinned is not held_source(analysis)
        assert [request.population for _, request in acquired_sources] == ["assigned", "triggered"]
        assert all(source is pinned for source, _ in acquired_sources)
    finally:
        analysis.close()


@pytest.mark.integration
def test_custom_provider_forwarding_artifact_has_no_capture_guarantee(tmp_path):
    from increment.evidence import EvidenceProvider, evidence_provider
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    source = cast(Any, held_source(analysis))
    adapter = evidence_provider(source)

    class Provider:
        evidence = adapter.evidence
        context = adapter.context

        def __getattr__(self, name):
            return getattr(adapter, name)

        def moments(self, request):
            return adapter.moments(request)

        def breakout_method_moments(self, request):
            return adapter.breakout_method_moments(request)

        def unit_rows(self, request):
            return adapter.unit_rows(request)

        def method_fit(self, request):
            return adapter.method_fit(request)

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

    provider = Provider()
    assert isinstance(provider, EvidenceProvider)
    try:
        rows = run(provider, coherence="captured")
        assert rows.metadata is not None
        assert rows.metadata.coherence is None
        assert rows.metadata.effective_modes
        import json

        import narwhals as nw

        from increment.tables import estimates_to_readout, readout_table

        modes = dict(rows.metadata.effective_modes)
        frame = nw.from_native(rows.to_frame(backend="polars"), eager_only=True)
        assert frame["coherence"].is_null().to_list() == [True] * len(rows)
        assert frame["provider_owned"].to_list() == [True] * len(rows)
        assert [json.loads(value) for value in frame["effective_modes"].to_list()] == [modes] * len(
            rows
        )
        exported = estimates_to_readout(rows)
        assert all(row["coherence"] is None for row in exported)
        assert all(row["provider_owned"] is True for row in exported)
        assert all(json.loads(row["effective_modes"]) == modes for row in exported)
        pytest.importorskip("coeftable")
        assert "Coherence: provider-owned" in readout_table(rows).gt().as_raw_html()
    finally:
        analysis.close()


@pytest.mark.integration
def test_triggered_native_wrapper_preserves_physical_stream_inventory(tmp_path):
    from increment import EvidenceScope
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.analysis_factory import held_source

    analysis, _ = definitions_source(
        tmp_path, trigger=True, sql_exposures=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    triggered = source.triggered_source()
    context = EvidenceExecutionContext(
        native_scope_provider(triggered),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            population="triggered",
        ),
        run_id="triggered-inventory",
    )
    try:
        context.open()
        assert {"events", "exposure:assignment", "exposure:trigger"} <= set(context.effective_modes)
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_custom_provider_preserves_snapshot_after_update(tmp_path):
    from increment import EvidenceScope
    from increment.evidence import EvidenceProvider, evidence_provider
    from increment.kernel.context import EvidenceExecutionContext

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    adapter = evidence_provider(source)

    class Provider:
        context = source.context
        evidence = source.evidence

        def __init__(self):
            self.requests = []

        def moments(self, request):
            return adapter.moments(request)

        def breakout_method_moments(self, request):
            return adapter.breakout_method_moments(request)

        def unit_rows(self, request):
            return adapter.unit_rows(request)

        def method_fit(self, request):
            return adapter.method_fit(request)

        def compliance(self, request):
            return adapter.compliance(request)

        def compliance_series(self, request):
            return adapter.compliance_series(request)

        def counts(self, request):
            return adapter.counts(request)

        def capture_sequential(self, request):
            return adapter.capture_sequential(request)

        def scope(self, request):
            self.requests.append(request)
            return adapter.scope(request)

    provider = Provider()
    assert isinstance(provider, EvidenceProvider)
    context = EvidenceExecutionContext(
        provider,
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            snapshot=True,
        ),
        run_id="custom-snapshot",
    )
    try:
        context.open()
        assert provider.requests[0].snapshot is True
        source._con.raw_sql(
            "UPDATE raw_events SET revenue = revenue + 100 "
            "WHERE event = 'purchase' AND group_id IS NULL"
        )
        rows = run(provider, execution_context=context)
        assert rows
        assert rows[0].require_lift().value == pytest.approx(2.0)
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_analysis_run_exposes_live_coherence_when_temp_tables_are_denied(tmp_path, monkeypatch):
    from increment.errors import CodedError
    from tests.analysis_factory import held_source

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    con = source._con
    create_table = con.create_table

    def deny_temp(name, expr, **kwargs):
        if kwargs.get("temp"):
            raise PermissionError("TEMP creation denied")
        return create_table(name, expr, **kwargs)

    monkeypatch.setattr(con, "create_table", deny_temp)
    try:
        with pytest.raises(CodedError) as error:
            analysis.run()
        assert error.value.code == "query.session.snapshot.materialization_failed"

        rows = analysis.run(coherence="live")
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence == "live"
        assert set(rows.metadata.effective_modes.values()) == {"live"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_frame_panel_context_snapshots_mutable_pandas_panel():
    import narwhals as nw
    import pandas as pd

    from increment import Analysis, EvidenceScope
    from increment.frame import FramePanelSource
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.analysis_factory import held_source

    frame = pd.DataFrame(
        {
            "unit_id": ["c1", "c2", "t1", "t2"] * 2,
            "group_id": ["control", "control", "treatment", "treatment"] * 2,
            "ds": ["2025-01-01"] * 4 + ["2025-01-02"] * 4,
            "value": [9.0, 11.0, 24.0, 28.0] * 2,
        }
    )
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit_id",
        group="group_id",
        date="ds",
        control="control",
        metrics={"value": "mean"},
    )
    base = cast(Any, held_source(analysis))
    panel = nw.from_native(frame, eager_only=True)
    source = FramePanelSource(
        panel=panel,
        metrics=list(base._specs),
        control=base._control,
        experiment_id=base._experiment_id,
        design=base.design,
        plan=base.plan,
        group_counts=dict(base._group_counts),
        n_filled=base._n_filled,
        first_exposure=base._first_exposure,
        exposure=base._exposure,
        observation_end=base._observation_end,
        fact_max_ds=base._fact_max_ds,
        unassigned=base._unassigned,
        synthesised_metrics=base.context.metrics,
        source_frame=panel,
        unit_column="unit_id",
    )
    context = EvidenceExecutionContext(
        native_scope_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            snapshot=True,
        ),
        run_id="mutable-panel-snapshot",
    )
    try:
        context.open()
        frame.loc[frame["group_id"] == "treatment", "value"] += 100.0
        rows = run(source, execution_context=context)
        assert rows[0].require_lift().value == pytest.approx(1.6)
        assert rows.metadata is not None
        assert rows.metadata.coherence == "captured"
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_frame_panel_context_snapshots_mutable_polars_panel():
    import narwhals as nw
    import polars as pl

    from increment import Analysis, EvidenceScope
    from increment.frame import FramePanelSource
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._provider import native_scope_provider
    from tests.analysis_factory import held_source

    frame = pl.DataFrame(
        {
            "unit_id": ["c1", "c2", "t1", "t2"] * 2,
            "group_id": ["control", "control", "treatment", "treatment"] * 2,
            "ds": ["2025-01-01"] * 4 + ["2025-01-02"] * 4,
            "value": [9.0, 11.0, 24.0, 28.0] * 2,
        }
    )
    analysis = Analysis.from_unit_panel(
        frame,
        unit="unit_id",
        group="group_id",
        date="ds",
        control="control",
        metrics={"value": "mean"},
    )
    base = cast(Any, held_source(analysis))
    panel = nw.from_native(frame, eager_only=True)
    source = FramePanelSource(
        panel=panel,
        metrics=list(base._specs),
        control=base._control,
        experiment_id=base._experiment_id,
        design=base.design,
        plan=base.plan,
        group_counts=dict(base._group_counts),
        n_filled=base._n_filled,
        first_exposure=base._first_exposure,
        exposure=base._exposure,
        observation_end=base._observation_end,
        fact_max_ds=base._fact_max_ds,
        unassigned=base._unassigned,
        synthesised_metrics=base.context.metrics,
        source_frame=panel,
        unit_column="unit_id",
    )
    context = EvidenceExecutionContext(
        native_scope_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            snapshot=True,
        ),
        run_id="mutable-polars-panel-snapshot",
    )
    try:
        context.open()
        frame.replace_column(
            frame.get_column_index("value"),
            pl.Series("value", [9.0, 11.0, 124.0, 128.0] * 2),
        )
        rows = run(source, execution_context=context)
        assert rows[0].require_lift().value == pytest.approx(1.6)
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_captured_analysis_keeps_unused_experiment_id_fact_column():
    from increment import AnalysisPlan
    from increment.semantics.models import Definitions
    from tests.analysis_factory import make_analysis
    from tests.parity_harness.cases import (
        _encouragement_defs_dict,
        _encouragement_rows_for_parity,
    )
    from tests.parity_harness.dataset import duckdb_connection

    definitions = _encouragement_defs_dict(AnalysisPlan(secondaries=["revenue"]))
    profiles = {
        **definitions["fact_sources"][0],
        "name": "profiles",
        "facts": [{"name": "profile_event", "column": "event"}],
        "properties": [],
    }
    definitions["fact_sources"].insert(0, profiles)
    disappeared = {
        **definitions["fact_sources"][0],
        "name": "disappeared",
        "sql": "SELECT * FROM raw_events WHERE event = 'not_a_real_event'",
        "facts": [{"name": "disappeared_event", "column": "event"}],
        "properties": [],
    }
    definitions["fact_sources"].insert(0, disappeared)
    definitions["fact_sources"][2]["facts"].append(
        {"name": "unused_experiment_id", "column": "experiment_id"}
    )
    definitions["experiments"].append({**definitions["experiments"][0], "name": "exp2"})
    base_rows = _encouragement_rows_for_parity()
    exposure_groups = {
        row["user_id"]: row["group_id"] for row in base_rows if row["event"] == "exposure"
    }
    rows = []
    for experiment_id, control_values, treatment_values in (
        ("exp", (9.0, 11.0), (24.0, 28.0)),
        ("exp2", (90.0, 110.0), (200.0, 220.0)),
    ):
        for original in base_rows:
            row = dict(original)
            original_user = row["user_id"]
            row["user_id"] = f"{experiment_id}_{original_user}"
            row["experiment_id"] = experiment_id
            if row["event"] == "purchase" and row["revenue"] != 0:
                arm = exposure_groups[original_user]
                values = control_values if arm == "control" else treatment_values
                row["revenue"] = values[int(original_user[1:]) % 2]
            rows.append(row)

    con = duckdb_connection(rows)
    analysis = make_analysis(
        con,
        Definitions.model_validate(definitions),
        experiment="exp",
    )
    source = cast(Any, held_source(analysis))
    try:
        live = run(source, coherence="live")
        captured = analysis.run()
        live_rows = lift_rows(live)
        captured_rows = lift_rows(captured)
        live_row = next(
            row for row in live_rows if row.metric == "revenue" and row.estimand == "itt"
        )
        captured_row = next(
            row for row in captured_rows if row.metric == "revenue" and row.estimand == "itt"
        )
        assert live_row.require_lift().value == pytest.approx(1.6)
        assert captured_row.require_lift().value == pytest.approx(live_row.require_lift().value)
        assert captured.metadata is not None
        assert {"events", "profiles"} <= set(captured.metadata.effective_modes)
        assert "disappeared" not in captured.metadata.effective_modes
    finally:
        analysis.close()
        con.disconnect()
