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
def test_nonsequential_run_accepts_evidence_provider_wrapped_source(tmp_path):
    from increment.evidence import evidence_provider

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    try:
        rows = run(evidence_provider(source))
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence == "captured"
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
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    con = source._con
    con.create_table("mutable_summary", obj=con.to_pyarrow(source._summary))
    source._summary = con.table("mutable_summary")
    context = EvidenceExecutionContext(
        evidence_provider(source),
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
    from increment.evidence import MomentsRequest, evidence_provider, request_evidence
    from increment.kernel.context import EvidenceExecutionContext
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    metric = source.context.metrics[0]
    context = EvidenceExecutionContext(
        evidence_provider(source),
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
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext
    from tests.source_conformance import moments_source

    source = moments_source()
    context = EvidenceExecutionContext(
        evidence_provider(source),
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
def test_daily_and_asof_collections_keep_distinct_scope_identity_and_mode(tmp_path):
    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    try:
        daily = analysis.run_daily()
        asof = analysis.run_asof()
        assert daily and asof
        assert daily.metadata is not None
        assert asof.metadata is not None
        assert daily.metadata.coherence == "captured"
        assert asof.metadata.coherence == "captured"
        assert (
            daily.metadata.cells[0].source_snapshot_id != asof.metadata.cells[0].source_snapshot_id
        )
        daily_identities = {cell.identity for cell in daily.as_cells()}
        asof_identities = {cell.identity for cell in asof.as_cells()}
        assert daily_identities.isdisjoint(asof_identities)
        assert {identity.scope.view for identity in daily_identities} == {"daily"}
        assert {identity.scope.view for identity in asof_identities} == {"asof"}
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
@pytest.mark.parametrize("view", ("daily", "asof"))
def test_artifact_day_axis_keeps_captured_metadata(tmp_path, view):
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    try:
        collection = analysis.run_daily() if view == "daily" else analysis.run_asof()
        assert collection.metadata is not None
        assert collection.metadata.coherence == "captured"
        assert collection.metadata.effective_modes
        assert set(collection.metadata.effective_modes.values()) == {"captured"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_artifact_public_run_keeps_captured_metadata_when_live_is_requested(tmp_path):
    from tests.analysis_factory import held_source
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    try:
        collection = run(cast(Any, held_source(analysis)), coherence="live")
        assert collection.metadata is not None
        assert collection.metadata.coherence == "captured"
        assert collection.metadata.effective_modes
        assert set(collection.metadata.effective_modes.values()) == {"captured"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_clustered_day_axis_refuses_before_opening_context(tmp_path, monkeypatch):
    from increment.errors import CodedError

    analysis, _ = definitions_source(tmp_path, cluster=True, return_analysis=True)
    source = cast(Any, held_source(analysis))
    original = source.readout_snapshot
    reads = 0

    def count_snapshot(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "readout_snapshot", count_snapshot)
    try:
        with pytest.raises(CodedError):
            analysis.run_daily()
        assert reads == 0
    finally:
        analysis.close()


@pytest.mark.integration
@pytest.mark.parametrize("operation", ("run_daily", "run_asof"))
def test_standalone_value_read_skips_separately_sourced_cuped_covariate(
    tmp_path, monkeypatch, operation
):
    import datetime as dt

    import yaml

    from increment import Analysis
    from tests.covariate_cases import categorical_defs_and_con

    path, con, units = categorical_defs_and_con(tmp_path)
    definitions = yaml.safe_load(path.read_text())
    experiment = definitions["experiments"][0]
    experiment.pop("design")
    experiment["n_pre_periods"] = 14
    experiment["plan"] = {
        "primary": {
            "metric": "revenue",
            "decision_method": {
                "name": "linear",
                "variance_reduction": "cuped_multicovariate",
                "covariates": [
                    {
                        "name": "region",
                        "kind": "categorical",
                        "property": "region",
                        "source": "profiles",
                    }
                ],
            },
        }
    }
    definitions["fact_sources"].append(
        {
            "name": "profiles",
            "sql": "SELECT * FROM profiles",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "profile", "column": None}],
            "properties": [
                {"name": "region", "column": "region", "dtype": "string", "as_of": "pre_exposure"}
            ],
        }
    )
    con.create_table(
        "profiles",
        obj=[
            {
                "user_id": unit,
                "ts": dt.datetime(2025, 7, 20),
                "region": values["region"],
                "event": "profile",
            }
            for unit, values in units.items()
        ],
    )
    path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions("cat_test", path, con, store="none")
    source = cast(Any, held_source(analysis))
    original = source._get_fact_table
    source_reads: list[str] = []

    def count_source(fact_source):
        source_reads.append(fact_source.name)
        return original(fact_source)

    monkeypatch.setattr(source, "_get_fact_table", count_source)
    con.drop_table("profiles")
    try:
        rows = getattr(analysis, operation)(metrics=["revenue"])
        assert rows
        assert "profiles" not in source_reads
        assert rows.metadata is not None
        assert "profiles" not in rows.metadata.effective_modes
    finally:
        analysis.close()


@pytest.mark.parametrize("missing_history_source", [None, "profiles_a", "profiles_b"])
@pytest.mark.integration
def test_captured_run_pins_each_source_for_same_property_covariates(
    tmp_path, missing_history_source
):
    import datetime as dt

    import yaml

    from increment import Analysis
    from tests.covariate_cases import categorical_defs_and_con

    path, con, units = categorical_defs_and_con(tmp_path)
    definitions = yaml.safe_load(path.read_text())
    experiment = definitions["experiments"][0]
    experiment.pop("design")
    experiment["n_pre_periods"] = 14
    experiment["plan"] = {
        "primary": {
            "metric": "revenue",
            "decision_method": {
                "name": "linear",
                "variance_reduction": "cuped_multicovariate",
                "covariates": [
                    {
                        "name": "region_a",
                        "kind": "categorical",
                        "property": "region",
                        "source": "profiles_a",
                    },
                    {
                        "name": "region_b",
                        "kind": "categorical",
                        "property": "region",
                        "source": "profiles_b",
                    },
                ],
            },
        }
    }
    for name in ("profiles_a", "profiles_b"):
        definitions["fact_sources"].append(
            {
                "name": name,
                "sql": f"SELECT * FROM {name}",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": f"{name}_marker", "column": None}],
                "properties": [
                    {
                        "name": "region",
                        "column": "region",
                        "dtype": "string",
                        "as_of": "pre_exposure",
                    }
                ],
            }
        )
        con.create_table(
            name,
            obj=[
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 7, 20),
                    "region": (
                        values["region"]
                        if name == "profiles_a"
                        else ("east", "west", "north")[int(unit[1:]) % 3]
                    ),
                    "event": "profile",
                }
                for unit, values in units.items()
            ],
        )
    if missing_history_source is not None:
        con.raw_sql(f"DELETE FROM {missing_history_source}")
    path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions("cat_test", path, con, store="none")
    try:
        captured_rows = analysis.run(coherence="captured")
        (captured,) = captured_rows
        from increment.estimation.results import LiftEstimate

        assert isinstance(captured, LiftEstimate)
        if missing_history_source is not None:
            assert captured.lift is None
            return
        live_rows = analysis.run(coherence="live")
        (live,) = live_rows
        assert isinstance(live, LiftEstimate)
        assert captured.require_lift().value is not None
        assert live.require_lift().value is not None
        assert captured.require_lift().value == pytest.approx(live.require_lift().value)
        assert captured_rows.metadata is not None
        assert {"profiles_a", "profiles_b"} <= set(captured_rows.metadata.effective_modes)
    finally:
        analysis.close()


@pytest.mark.integration
def test_captured_run_keeps_unbounded_properties_for_shared_sql(tmp_path):
    import datetime as dt

    import yaml

    from increment import Analysis
    from tests.covariate_cases import categorical_defs_and_con

    path, con, units = categorical_defs_and_con(tmp_path)
    definitions = yaml.safe_load(path.read_text())
    experiment = definitions["experiments"][0]
    experiment.pop("design")
    experiment["n_pre_periods"] = 14
    experiment["plan"] = {
        "primary": {
            "metric": "revenue",
            "decision_method": {
                "name": "scalar",
                "variance_reduction": "cuped",
                "covariates": [
                    {
                        "name": "tenure",
                        "kind": "continuous",
                        "property": "tenure",
                        "source": "profiles",
                    }
                ],
            },
        }
    }
    definitions["fact_sources"] = [
        {
            "name": "events",
            "sql": "SELECT * FROM exposure_rows",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "exposure", "column": None}],
            "properties": [],
        },
        {
            "name": "z_revenue",
            "sql": "SELECT * FROM outcome_rows",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "purchase", "column": "revenue"}],
            "properties": [],
        },
        {
            "name": "profiles",
            "sql": "SELECT * FROM outcome_rows",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "profile_marker", "column": None}],
            "properties": [
                {
                    "name": "tenure",
                    "column": "tenure",
                    "dtype": "float",
                    "as_of": "pre_exposure",
                }
            ],
        },
    ]
    con.create_table(
        "exposure_rows",
        obj=[
            {
                "user_id": unit,
                "ts": dt.datetime(2025, 8, 1),
                "event": "exposure",
                "experiment_id": "cat_test",
                "group_id": values["variant"],
            }
            for unit, values in units.items()
        ],
    )
    outcome_rows = []
    for index, (unit, values) in enumerate(units.items()):
        outcome_rows.extend(
            [
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 6, 15),
                    "event": "profile",
                    "revenue": 0.0,
                    "tenure": values["tenure"],
                },
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 8, 3),
                    "event": "purchase",
                    "revenue": values["revenue"],
                },
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 8, 10),
                    "event": "purchase",
                    "revenue": float(index % 2),
                },
            ]
        )
    con.create_table("outcome_rows", obj=outcome_rows)
    path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions("cat_test", path, con, store="none")
    try:
        from increment.estimation.results import LiftEstimate

        live_rows = analysis.run(coherence="live")
        (live,) = live_rows
        assert isinstance(live, LiftEstimate)
        assert live.lift is not None, live.model_dump()
        captured_rows = analysis.run(coherence="captured")
        (captured,) = captured_rows
        assert isinstance(captured, LiftEstimate)
        assert captured.require_lift().value == pytest.approx(live.require_lift().value)
    finally:
        analysis.close()


@pytest.mark.integration
def test_snapshot_ignores_ambiguous_covariate_on_unselected_metric(tmp_path):
    import datetime as dt

    import yaml

    from increment import Analysis
    from increment.errors import CodedError
    from tests.covariate_cases import categorical_defs_and_con

    path, con, units = categorical_defs_and_con(tmp_path)
    definitions = yaml.safe_load(path.read_text())
    definitions["metrics"].append(
        {
            "name": "other_revenue",
            "type": "mean",
            "entity": "user_id",
            "fact": "purchase",
            "aggregation": "sum",
            "window_days": 7,
        }
    )
    experiment = definitions["experiments"][0]
    experiment.pop("design")
    experiment["n_pre_periods"] = 14
    experiment["plan"] = {
        "primary": "revenue",
        "secondaries": [
            {
                "metric": "other_revenue",
                "decision_method": {
                    "name": "linear",
                    "variance_reduction": "cuped_multicovariate",
                    "covariates": [{"name": "region", "kind": "categorical"}],
                },
            }
        ],
    }
    definitions["fact_sources"].append(
        {
            "name": "profiles",
            "sql": "SELECT * FROM profiles",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "profile", "column": None}],
            "properties": [
                {
                    "name": "region",
                    "column": "region",
                    "dtype": "string",
                    "as_of": "pre_exposure",
                }
            ],
        }
    )
    con.create_table(
        "profiles",
        obj=[
            {
                "user_id": unit,
                "ts": dt.datetime(2025, 7, 20),
                "region": values["region"],
                "event": "profile",
            }
            for unit, values in units.items()
        ],
    )
    path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions("cat_test", path, con, store="none")
    try:
        with pytest.raises(CodedError) as ambiguous:
            analysis.run(metrics=["other_revenue"])
        assert ambiguous.value.code == "source.native.covariate_ambiguous"

        expected_counts = {
            group: sum(values["variant"] == group for values in units.values())
            for group in ("control", "treatment")
        }
        assert analysis.unit_counts() == expected_counts

        assert analysis.allocation_history().num_rows > 0

        def read_snapshot(pinned: Analysis) -> Any:
            return pinned.unit_counts()

        assert analysis.dashboard_snapshot(read_snapshot, metrics=[]) == expected_counts
        result = analysis.dashboard_snapshot(read_snapshot, metrics=[analysis.metrics[0]])
        assert result == expected_counts
    finally:
        analysis.close()


@pytest.mark.integration
def test_public_breakout_override_does_not_change_sibling_daily_lift():
    from increment import Method
    from tests.parity_harness.cases import PARITY_CASES

    case = next(case for case in PARITY_CASES if case.id == "breakout_store_dimension")
    analysis = case.build["from_definitions"]()
    try:
        breakout = analysis.run_breakout(
            metrics=("revenue_cuped",),
            decision_method=Method.unadjusted(name="breakout-unadjusted"),
        )
        daily_lift = analysis.run_daily_lift(metrics=("revenue_cuped",))
        assert breakout
        assert daily_lift
        assert "breakout-unadjusted" in {row.method for row in breakout}
        assert "breakout-unadjusted" not in {row.method for row in daily_lift}
    finally:
        analysis.close()


@pytest.mark.integration
def test_standalone_frame_panel_readouts_own_mode_metadata():
    from tests.source_conformance import frame_panel_analysis

    analysis = frame_panel_analysis()
    try:
        for operation in ("run_daily", "run_daily_lift", "run_asof_lift", "run_breakout"):
            rows = getattr(analysis, operation)()
            assert rows
            assert rows.metadata is not None
            assert rows.metadata.effective_modes
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
def test_custom_provider_keeps_provider_qualification_under_captured_request():
    from tests.test_readouts_encouragement import METRIC, FakeMomentSource, _design, _rows

    source = FakeMomentSource(
        _rows(n=100),
        metrics=[METRIC],
        capabilities={"total"},
        design=_design(),
    )
    rows = run(source, coherence="captured")
    assert rows.metadata is not None
    assert rows.metadata.coherence is None
    assert set(rows.metadata.effective_modes.values()) == {"provider"}


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
    from increment.evidence import evidence_provider
    from increment.kernel.context import EvidenceExecutionContext
    from tests.analysis_factory import held_source

    analysis, _ = definitions_source(
        tmp_path, trigger=True, sql_exposures=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    triggered = source.triggered_source()
    context = EvidenceExecutionContext(
        evidence_provider(triggered),
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
@pytest.mark.parametrize("coherence", ("captured", "live"))
def test_native_scope_result_binds_breakout_to_captured_source(tmp_path, coherence):
    from increment import BreakoutIdentity, EvidenceScope
    from increment.evidence import evidence_provider

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    identity = BreakoutIdentity("country", None)
    try:
        scope = EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            breakouts=(identity,),
            snapshot=True,
            execution=True,
        )
        owner = evidence_provider(source)
        result = owner.scope(scope)
        assert result.breakout_streams == {identity: "events"}

        from increment.kernel.context import EvidenceExecutionContext

        context = EvidenceExecutionContext(
            owner, scope, run_id="typed-breakout-binding", coherence=coherence
        )
        context.open()
        try:
            context.validate_breakouts((identity,))
        finally:
            context.close()
    finally:
        analysis.close()


@pytest.mark.integration
def test_frame_panel_context_snapshots_mutable_pandas_panel():
    import narwhals as nw
    import pandas as pd

    from increment import Analysis, EvidenceScope
    from increment.evidence import evidence_provider
    from increment.frame import FramePanelSource
    from increment.kernel.context import EvidenceExecutionContext
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
        evidence_provider(source),
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
    from increment.evidence import evidence_provider
    from increment.frame import FramePanelSource
    from increment.kernel.context import EvidenceExecutionContext
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
        evidence_provider(source),
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
def test_undeclared_breakout_source_refuses_before_the_read(tmp_path, monkeypatch):
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
            configs=tuple(source.context.configs),
            execution=True,
            include_breakouts=True,
        ),
        run_id="undeclared-breakout-identity",
    )
    captures = 0
    original = source.readout_snapshot

    def count_capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "readout_snapshot", count_capture)
    try:
        context.open()
        opened = captures
        with pytest.raises(CodedError) as refused:
            analysis._run_breakout(_execution_context=context)
        assert refused.value.code == "kernel.context.undeclared_stream"
        assert captures == opened
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


@pytest.mark.integration
def test_public_unit_summary_breakout_keeps_declared_segments():
    from tests.parity_harness.cases import PARITY_CASES

    case = next(case for case in PARITY_CASES if case.id == "breakout_store_dimension")
    analysis = case.build["from_unit_summary"]()
    try:
        breakout = analysis.run_breakout()
        assert len(breakout) == 15
        assert breakout.metadata is not None
        assert breakout.metadata.coherence == "captured"
        assert set(breakout.metadata.effective_modes.values()) == {"captured"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_windowed_daily_scope_does_not_extend_past_metric_window():
    from datetime import date

    from tests.parity_harness import matrix, matrix_cases

    cell = matrix.Cell("windowed_mean", "daily", "sequential", "utc", "error")
    disposition = matrix.classify(cell)
    case = matrix_cases.build_case(cell, "values", disposition.outcomes("values"))
    expected_days = {date(2025, 1, 10), date(2025, 1, 11)}
    for ingress in ("from_definitions", "from_unit_panel"):
        analysis = case.build[ingress]()
        try:
            rows = analysis.run_daily()
            assert {row.ds for row in rows} == expected_days
            assert len(rows) == 4
        finally:
            analysis.close()


@pytest.mark.integration
def test_daily_missing_date_keeps_exact_segment_arm_cell_without_extra_event(tmp_path):
    from datetime import date

    analysis, _ = definitions_source(tmp_path, return_analysis=True, window_days=2)
    try:
        with pytest.warns():
            rows = analysis.run_daily(metrics=["revenue"], dimension="country")
        missing = [
            row
            for row in rows
            if row.ds == date(2026, 1, 2)
            and row.dimension_value == "US"
            and row.group_id == "treatment"
        ]
        assert len(missing) == 1
        assert missing[0].value is None
        assert missing[0].unavailable == "few_units"
    finally:
        analysis.close()


@pytest.mark.integration
def test_dimensioned_asof_daily_lift_retains_registered_empty_roster_member():
    from datetime import date

    from increment import AlwaysValid, SequentialCell, SequentialRegistration
    from increment.breakout.estimates import run_daily_lift
    from increment.semantics.models import ConversionMetric
    from tests.sequential_cases import capture, registration

    base = registration("bernoulli")
    levels = ("US", "CA", "ABSENT")
    registered = SequentialRegistration.model_validate(
        {
            **base.model_dump(),
            "roster": tuple(
                SequentialCell(
                    metric="outcome",
                    group_id="treatment",
                    segment=(("country", level),),
                    family=True,
                )
                for level in levels
            ),
        }
    )
    observations = []
    for country in ("US", "CA"):
        for index in range(24):
            for arm, value in (
                ("control", index % 2),
                ("treatment", (index + 1) % 2),
            ):
                observations.append(
                    {
                        "unit_id": f"{country}-{index}-{arm}",
                        "group_id": arm,
                        "segments": {"country": country},
                        "values": {"outcome": value},
                    }
                )
    snapshot = capture(registered, observations).model_copy(
        update={"reveal_cursor": date(2025, 1, 1)}
    )
    rows = run_daily_lift(
        snapshot,
        [ConversionMetric(name="outcome", entity="user", fact="events")],
        control_group="control",
        dimension="country",
        view="asof",
        inference=AlwaysValid(registration=registered),
    )
    assert {row.dimension_value for row in rows} == set(levels)
    absent = next(row for row in rows if row.dimension_value == "ABSENT")
    assert absent.sequential_result is not None
    assert absent.sequential_result.checkpoint.status == "missing"


@pytest.mark.integration
def test_dimensioned_daily_lift_discovers_unregistered_segment_before_estimability():
    from datetime import date

    from increment.breakout.estimates import run_daily_lift
    from increment.estimation.armstats import centered_row_from_raw_sums
    from increment.semantics.models import MeanMetric

    n = 12.0
    mean = 2.0
    total = n * mean
    summary = centered_row_from_raw_sums(
        {
            "ds": date(2025, 1, 1),
            "experiment_id": "exp",
            "metric": "revenue",
            "group_id": "treatment",
            "n": n,
            "sum_y": total,
            "sum_y2": 11.0 + total * total / n,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )
    summary["country"] = "new-segment"
    rows = run_daily_lift(
        [summary],
        [MeanMetric(name="revenue", entity="user", fact="purchase")],
        control_group="control",
        dimension="country",
    )
    treatment = [row for row in rows if row.group_id == "treatment"]
    assert len(treatment) == 1
    assert treatment[0].dimension_value == "new-segment"
    assert treatment[0].unavailable == "no_control_arm"


@pytest.mark.integration
def test_breakout_override_preserves_sibling_daily_lift_configuration(tmp_path):
    from datetime import date

    import yaml

    from increment import Analysis, Method
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import _DEFINITIONS_PLAN

    rows = []
    for treatment, arm, prefix in ((0, "control", "c"), (1, "treatment", "t")):
        for index in range(20):
            x = index + 2 * treatment
            user_id = f"{prefix}{index}"
            store_id = ds.store_for(index)
            rows.extend(
                (
                    ds._row(
                        user_id,
                        ds._EXPOSURE_AT,
                        "exposure",
                        group_id=arm,
                        experiment_id="exp",
                        store_id=store_id,
                    ),
                    ds._row(
                        user_id,
                        ds._PRE_PURCHASE_AT,
                        "purchase",
                        store_id=store_id,
                        revenue=float(x + 1),
                    ),
                    ds._row(
                        user_id,
                        ds._PURCHASE_AT,
                        "purchase",
                        store_id=store_id,
                        revenue=float(10 + 2 * x + (index % 3) + 3 * treatment),
                    ),
                    ds._row(
                        user_id,
                        ds._SESSION_AT,
                        "session_end",
                        store_id=store_id,
                        sess=1,
                    ),
                    ds._row(
                        user_id,
                        ds._FRESHNESS_PAD_AT,
                        "purchase",
                        store_id=store_id,
                        revenue=0.0,
                    ),
                )
            )
    connection = ds.duckdb_connection(rows)
    definitions = ds.definitions_dict(plan=_DEFINITIONS_PLAN, breakout=True)
    definitions_path = tmp_path / "cuped_breakout_override.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions))
    analysis = Analysis.from_definitions(
        "exp", definitions_path, cast(Any, connection), store="none"
    )
    try:
        standalone_daily = analysis.run_daily_lift(metrics=("revenue_cuped",))
        standalone_asof = analysis.run_asof_lift(metrics=("revenue_cuped",))
        analysis.run_breakout(
            metrics=("revenue_cuped",),
            decision_method=Method.unadjusted(name="breakout-only"),
        )
        scoped = {
            "daily_lift": analysis.run_daily_lift(metrics=("revenue_cuped",)),
            "asof_lift": analysis.run_asof_lift(metrics=("revenue_cuped",)),
        }
        assert standalone_daily and standalone_asof
        for name, standalone in (("daily_lift", standalone_daily), ("asof_lift", standalone_asof)):
            baseline = [
                row
                for row in standalone
                if row.metric == "revenue_cuped" and row.ds == date(2025, 1, 10)
            ]
            actual = [
                row
                for row in scoped[name]
                if row.metric == "revenue_cuped" and row.ds == date(2025, 1, 10)
            ]
            assert len(baseline) == len(actual) == 2
            assert {row.method for row in baseline} == {"cuped", "unadjusted"}
            cuped = next(row for row in baseline if row.method == "cuped")
            assert cuped.require_lift().value == pytest.approx(0.09355184, abs=1e-8)
            assert [row.require_lift().value for row in actual] == pytest.approx(
                [row.require_lift().value for row in baseline]
            )
            assert {row.method for row in actual} == {row.method for row in baseline}
    finally:
        analysis.close()
        connection.disconnect()


@pytest.mark.integration
def test_unsegmented_day_axis_reads_do_not_require_separate_breakout_sources(tmp_path):
    from tests.test_dashboard_capture import _workspace

    with _workspace(tmp_path, units=6, breakout_sources=("profiles",)) as (con, analysis):
        con.drop_table("profiles", database="analytics")
        for operation in ("run_daily", "run_asof", "run_daily_lift", "run_asof_lift"):
            rows = getattr(analysis, operation)(metrics=["revenue_per_user"])
            assert rows


@pytest.mark.integration
def test_breakout_scope_excludes_unselected_metric_fact_source(tmp_path):
    import ibis
    import yaml

    from increment import Analysis
    from tests.covariate_cases import _defs_yaml, _unit_rows

    event_rows = []
    unused_rows = []
    for index, group in enumerate(["control"] * 10 + ["treatment"] * 10):
        unit = f"u{index}"
        event_rows.extend(
            {**row, "segment": f"s{index % 2}"}
            for row in _unit_rows("cov_test", unit, group, float(index), float(index + 1))
        )
        unused_rows.append(
            {
                "user_id": unit,
                "ts": __import__("datetime").datetime(2025, 8, 3),
                "event": "unused",
                "unused_value": float(index),
            }
        )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows)
    con.create_table("unused_events", obj=unused_rows)
    document = yaml.safe_load(_defs_yaml("events", "cov_test", observational=False))
    document["fact_sources"][0]["properties"].append(
        {"name": "segment", "column": "segment", "dtype": "string", "as_of": "static"}
    )
    document["fact_sources"].append(
        {
            "name": "unused",
            "sql": "SELECT * FROM unused_events",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "unused_fact", "column": "unused_value"}],
        }
    )
    document["fact_sources"].append(
        {
            "name": "account_events",
            "sql": document["fact_sources"][0]["sql"],
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [],
            "properties": [
                {"name": "segment", "column": "segment", "dtype": "string", "as_of": "static"}
            ],
        }
    )
    document["metrics"].append(
        {
            "type": "mean",
            "name": "unselected",
            "entity": "user_id",
            "fact": "unused_fact",
            "aggregation": "sum",
            "window_days": 7,
        }
    )
    document["experiments"][0]["breakouts"] = [{"property": "segment", "source": "events"}]
    path = tmp_path / "selective_breakout.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    analysis = Analysis.from_definitions("cov_test", path, con)
    try:
        con.drop_table("unused_events")
        rows = analysis.run_breakout(metrics=["revenue"])
        assert rows
        assert {row.metric for row in rows} == {"revenue"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_daily_dimension_does_not_capture_unselected_dimension_source(tmp_path):
    import ibis
    import yaml

    from increment import Analysis
    from tests.covariate_cases import _defs_yaml, _unit_rows

    event_rows = []
    other_rows = []
    for index, group in enumerate(["control"] * 10 + ["treatment"] * 10):
        unit = f"u{index}"
        event_rows.extend(
            {**row, "country": "US" if index % 2 else "CA"}
            for row in _unit_rows("cov_test", unit, group, float(index), float(index + 1))
        )
        other_rows.append(
            {
                "user_id": unit,
                "ts": __import__("datetime").datetime(2025, 7, 20),
                "device": "mobile",
            }
        )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows)
    con.create_table("other_events", obj=other_rows)
    document = yaml.safe_load(_defs_yaml("events", "cov_test", observational=False))
    document["fact_sources"][0]["properties"].append(
        {"name": "country", "column": "country", "dtype": "string", "as_of": "static"}
    )
    document["fact_sources"].append(
        {
            "name": "other",
            "sql": "SELECT * FROM other_events",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [],
            "properties": [
                {"name": "device", "column": "device", "dtype": "string", "as_of": "static"}
            ],
        }
    )
    document["experiments"][0]["breakouts"] = [
        {"property": "country", "source": "events"},
        {"property": "device", "source": "other", "skip_missing": True},
    ]
    path = tmp_path / "selective_daily.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    analysis = Analysis.from_definitions("cov_test", path, con)
    try:
        con.drop_table("other_events")
        rows = analysis.run_daily(metrics=["revenue"], dimension="country")
        assert rows
        assert {row.dimension for row in rows} == {"country"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_standalone_daily_lift_rejects_sequential_plan_before_snapshot(tmp_path, monkeypatch):
    from increment.errors import UnsupportedRequestError
    from tests.sequential_cases import registered_native

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    analysis = registered_native(analysis, public_mean=True)
    source = cast(Any, held_source(analysis))
    snapshots = 0
    original_snapshot = source.readout_snapshot

    def count_snapshot(*args, **kwargs):
        nonlocal snapshots
        snapshots += 1
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(source, "readout_snapshot", count_snapshot)
    try:
        with pytest.raises(UnsupportedRequestError) as raised:
            analysis.run_daily_lift()
        assert raised.value.code == "readout.inference.disjoint_slices"
        assert snapshots == 0
    finally:
        analysis.close()


@pytest.mark.integration
def test_undeclared_breakout_dimension_refuses_before_the_read(tmp_path, monkeypatch):
    from increment import EvidenceExecutionContext, EvidenceScope
    from increment.errors import CodedError
    from increment.evidence import evidence_provider

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    owner = evidence_provider(source)
    context = EvidenceExecutionContext(
        owner,
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            breakouts=(),
            execution=True,
        ),
        run_id="undeclared-breakout",
    )
    captures = 0
    original = source.readout_snapshot

    def count_capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "readout_snapshot", count_capture)
    try:
        context.open()
        opened = captures
        with pytest.raises(CodedError) as refused:
            analysis._run_breakout(_execution_context=context)
        assert refused.value.code == "kernel.context.undeclared_stream"
        assert captures == opened
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_line2_owner_reuses_one_capture_for_primary_and_auxiliary_readouts(tmp_path, monkeypatch):
    from increment.readouts import run_scoped

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    captures = 0
    original = source.readout_snapshot

    def count_capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(source, "readout_snapshot", count_capture)
    try:
        with pytest.warns():
            results = run_scoped(analysis, breakout_correction="none")
        assert set(results) == {"run", "breakout", "daily", "daily_lift", "asof", "asof_lift"}
        assert captures == 1
        for collection in results.values():
            assert collection.metadata is not None
            assert collection.metadata.coherence == "captured"
    finally:
        analysis.close()


def test_sequential_run_pins_the_registered_family_and_records_mode():
    from increment.frame import MetricSpec
    from tests.test_sequential_public_sources import _analysis, _frame, gaussian_plan

    specs = [
        MetricSpec(name="outcome", type="conversion"),
        MetricSpec(name="sibling", type="conversion"),
    ]
    frame = _frame([0, 1] * 24, [1, 1] * 24)
    frame["sibling"] = frame["outcome"]
    analysis = _analysis(
        frame,
        specs,
        gaussian_plan(specs, law="bernoulli", source_id="experiment", unit="unit", group="arm"),
    )
    try:
        rows = analysis.run(metrics=["outcome"])
        assert rows
        metadata = rows.metadata
        assert metadata is not None
        assert metadata.coherence == "captured"
        assert set(metadata.effective_modes.values()) == {"captured"}
        family_metrics = {
            member.metric for family in metadata.scope.families for member in family.members
        }
        assert {"outcome", "sibling"} <= family_metrics
    finally:
        analysis.close()


@pytest.mark.integration
def test_daily_missing_date_keeps_exact_segment_arm_cell(tmp_path):
    from datetime import date

    analysis, (_, con) = definitions_source(
        tmp_path, return_analysis=True, window_days=2, return_connection=True
    )
    try:
        con.raw_sql(
            "INSERT INTO raw_events (user_id, ts, event, group_id, country_code, revenue) "
            "VALUES ('u3', TIMESTAMP '2026-01-03 12:00:00', 'purchase', NULL, 'US', 4.0)"
        )
        with pytest.warns():
            rows = analysis.run_daily(metrics=["revenue"], dimension="country")
        missing = [
            row
            for row in rows
            if row.ds == date(2026, 1, 2)
            and row.dimension_value == "US"
            and row.group_id == "treatment"
        ]
        assert len(missing) == 1
        assert missing[0].value is None
        assert missing[0].unavailable == "few_units"
    finally:
        analysis.close()


def test_sequential_continuation_and_readout_share_run_owner():
    from increment.frame import MetricSpec
    from increment.readouts import run_scoped
    from tests.test_sequential_public_sources import _analysis, _frame, gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion")]
    frame = _frame([0, 1] * 24, [1, 1] * 24)
    analysis = _analysis(
        frame,
        specs,
        gaussian_plan(specs, law="bernoulli", source_id="experiment", unit="unit", group="arm"),
    )

    def two_looks():
        first = analysis.capture_sequential(finalized=True, freeze=("outcome",))
        second = analysis.capture_sequential(finalized=True, previous=first)
        rows = analysis.run()
        return first, second, rows

    try:
        first, second, rows = run_scoped(analysis, two_looks)
        second.verify_parent(first)
        assert second.frozen == first.frozen
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence == "captured"
        assert analysis.sequential_snapshot().prefix_id == second.prefix_id
    finally:
        analysis.close()


def test_sequential_rewritten_prefix_refuses_through_run_owner():
    from increment.errors import CodedError
    from increment.frame import MetricSpec
    from increment.readouts import run_scoped
    from tests.test_sequential_public_sources import _analysis, _frame, gaussian_plan

    specs = [MetricSpec(name="outcome", type="conversion")]
    plan = gaussian_plan(specs, law="bernoulli", source_id="experiment", unit="unit", group="arm")
    original = _analysis(_frame([0, 1] * 24, [1, 1] * 24), specs, plan)
    rewritten_frame = _frame([1, 0] * 24, [1, 1] * 24)
    rewritten = _analysis(rewritten_frame, specs, plan)
    previous = original.capture_sequential(finalized=True, freeze=("outcome",))
    try:
        with pytest.raises(CodedError) as refusal:
            run_scoped(
                rewritten,
                lambda: rewritten.capture_sequential(
                    finalized=True, previous=previous, freeze=("outcome",)
                ),
            )
        assert refusal.value.code == "sequential.continuation.rewrite"
    finally:
        original.close()
        rewritten.close()


@pytest.mark.integration
def test_borrowed_context_rejects_an_undeclared_population_before_read(tmp_path):
    from increment import EvidenceExecutionContext, EvidenceScope
    from increment.errors import CodedError
    from increment.evidence import evidence_provider
    from increment.readouts import run

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    source = cast(Any, held_source(analysis))
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            population="assigned",
            configs=tuple(source.context.configs),
            execution=True,
        ),
        run_id="undeclared-population",
    )
    try:
        context.open()
        with pytest.raises(CodedError) as refused:
            run(source, _population="triggered", execution_context=context)
        assert refused.value.code == "kernel.context.undeclared_stream"
        assert refused.value.context == {"stream": "population:triggered"}
    finally:
        context.close()
        analysis.close()


@pytest.mark.integration
def test_line_four_daily_readout_preserves_triggered_population(tmp_path):
    from increment.readouts import run_scoped

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    try:
        result = run_scoped(analysis, population="triggered", readouts=("daily",))["daily"]
        assert result
        assert {row.analysis_population for row in result} == {"triggered"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_typed_trigger_population_requests_are_rejected_before_reads():
    from increment import EvidenceExecutionContext, EvidenceScope
    from increment.errors import CodedError
    from increment.evidence import (
        ComplianceSeriesRequest,
        CountsRequest,
        evidence_provider,
    )
    from tests.test_compliance_summary import witness

    source = witness()
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            metrics=tuple(source.context.metrics),
            configs=tuple(source.context.configs),
            populations=("assigned",),
        ),
        run_id="typed-population-gate",
    )
    requests = (
        CountsRequest(kind="triggered"),
        CountsRequest(kind="unit", population="triggered"),
        ComplianceSeriesRequest(design=source.context.design, population="triggered"),
    )
    try:
        for request in requests:
            with pytest.raises(CodedError) as refused:
                context.validate_typed_request(request)
            assert refused.value.code == "kernel.context.undeclared_stream"
            assert refused.value.context == {"stream": "population:triggered"}
    finally:
        source.close()


@pytest.mark.integration
def test_metric_free_custom_compliance_retains_provider_mode():
    from increment import EvidenceExecutionContext, EvidenceScope
    from increment.evidence import evidence_provider
    from increment.readouts import run as readout_run
    from tests.test_readouts_encouragement import METRIC, FakeMomentSource, _design, _rows

    source = FakeMomentSource(
        _rows(n=200),
        metrics=[METRIC],
        capabilities={"total"},
        design=_design(),
    )
    context = EvidenceExecutionContext(
        evidence_provider(source),
        EvidenceScope(
            population="assigned",
            populations=("assigned",),
            uptake_facts=("help_click",),
            execution=True,
        ),
        run_id="metric-free-provider-compliance",
    )
    try:
        with context.evidence_scope():
            rows = readout_run(
                source,
                metrics=[],
                estimands=("compliance",),
                execution_context=context,
            )
        assert rows
        assert rows.metadata is not None
        assert rows.metadata.coherence is None
        assert rows.metadata.effective_modes
        assert set(rows.metadata.effective_modes.values()) == {"provider"}
        assert "provider" in rows.model_dump_json()
        assert "provider" in repr(rows)
    finally:
        source.close()


@pytest.mark.integration
def test_triggered_day_axis_borrows_the_run_capture(tmp_path):
    from increment.errors import CodedError
    from increment.readouts import run_scoped

    analysis, (_, connection) = definitions_source(
        tmp_path,
        trigger=True,
        sql_exposures=True,
        return_analysis=True,
        return_connection=True,
    )
    try:
        expected = analysis.run_daily(population="triggered")

        def mutate_after_run(_context):
            assert analysis.run()
            connection.raw_sql("DELETE FROM raw_events WHERE event = 'purchase'")

        try:
            actual = run_scoped(
                analysis,
                lambda: analysis.run_daily(population="triggered"),
                population="triggered",
                readouts=("run", "daily"),
                on_context=mutate_after_run,
            )
        except CodedError as refused:
            assert refused.code == "kernel.snapshot.integrity"
        else:
            expected_by_cell = {
                (row.metric, row.group_id, row.ds): (
                    row.n,
                    None if row.value is None else row.value.value,
                )
                for row in expected
            }
            actual_by_cell = {
                (row.metric, row.group_id, row.ds): (
                    row.n,
                    None if row.value is None else row.value.value,
                )
                for row in actual
            }
            assert actual_by_cell.keys() == expected_by_cell.keys()
            for identity, (n, value) in expected_by_cell.items():
                actual_n, actual_value = actual_by_cell[identity]
                assert actual_n == n
                assert actual_value == pytest.approx(value)
            assert {row.analysis_population for row in actual} == {"triggered"}
    finally:
        analysis.close()


@pytest.mark.integration
def test_compliance_only_day_axis_rejects_another_analysis_source():
    from increment.errors import CodedError
    from increment.readouts import run_scoped
    from tests.source_conformance import encouragement_panel_analysis

    owner = encouragement_panel_analysis()
    other = encouragement_panel_analysis()
    try:
        with pytest.raises(CodedError) as refused:
            run_scoped(
                owner,
                lambda: other.run_asof_lift(estimands=("compliance",)),
                population="assigned",
                readouts=("asof_lift",),
            )
        assert refused.value.code == "kernel.context.undeclared_stream"
        assert refused.value.context == {"stream": "source_ref"}
    finally:
        owner.close()
        other.close()


@pytest.mark.integration
def test_scoped_run_declares_both_triggered_populations(tmp_path):
    from increment.readouts import run_scoped

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    try:
        results = (
            run_scoped(analysis, readouts=("run",))["run"],
            run_scoped(analysis, lambda: analysis.run()),
        )
        for rows in results:
            assert {"assigned", "triggered"} <= {row.analysis_population for row in rows}
    finally:
        analysis.close()


@pytest.mark.integration
def test_scoped_run_result_supports_pickle_and_deepcopy(tmp_path):
    import copy
    import pickle

    from increment.readouts import run_scoped

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    try:
        rows = run_scoped(analysis, readouts=("run",))["run"]
        for restored in (pickle.loads(pickle.dumps(rows)), copy.deepcopy(rows)):
            assert restored.metadata is not None
            assert restored.metadata.effective_modes == rows.metadata.effective_modes
            assert tuple(restored) == tuple(rows)
    finally:
        analysis.close()


@pytest.mark.integration
def test_scoped_callback_uses_its_actual_encouragement_estimands():
    from increment.readouts import run_scoped
    from tests.source_conformance import encouragement_panel_analysis

    analysis = encouragement_panel_analysis()
    try:
        rows = run_scoped(
            analysis,
            lambda: analysis.run_daily_lift(estimands=("itt",)),
            population="assigned",
            readouts=("daily_lift",),
        )
        assert rows
        assert {row.estimand for row in rows} == {"itt"}
    finally:
        analysis.close()
