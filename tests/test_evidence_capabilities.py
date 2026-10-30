from __future__ import annotations

import copy
import pickle
from contextlib import contextmanager
from typing import Any, cast

import pytest

from increment import Analysis
from increment.evidence import (
    ComplianceRequest,
    ComplianceSummary,
    CountEvidence,
    CountsRequest,
    EvidenceScopeResult,
    MomentRows,
    MomentsRequest,
    UnitRows,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)
from tests.analysis_factory import held_source
from tests.source_conformance import (
    switchback_analysis,
    unit_day_artifact_analysis,
    unit_panel_table,
    unit_summary_table,
)


class _ForwardingProvider:
    def __init__(self, source):
        self.source = source
        self.scope_calls = 0

    @property
    def evidence(self):
        return self.source.evidence

    @property
    def context(self):
        return self.source.context

    def moments(self, request):
        return evidence_provider(self.source).moments(request)

    def unit_rows(self, request):
        return evidence_provider(self.source).unit_rows(request)

    def method_fit(self, request):
        return evidence_provider(self.source).method_fit(request)

    def breakout_method_moments(self, request):
        return evidence_provider(self.source).breakout_method_moments(request)

    def compliance(self, request):
        return evidence_provider(self.source).compliance(request)

    def compliance_series(self, request):
        return evidence_provider(self.source).compliance_series(request)

    def counts(self, request):
        return evidence_provider(self.source).counts(request)

    def capture_sequential(self, request):
        return evidence_provider(self.source).capture_sequential(request)

    @contextmanager
    def _scoped_source(self, request):
        with evidence_provider(self.source).scope(request).source as scoped:
            yield _ForwardingProvider(scoped)

    def scope(self, request):
        self.scope_calls += 1
        return EvidenceScopeResult(self._scoped_source(request), request)

    @contextmanager
    def readout_execution(self):
        yield

    @contextmanager
    def readout_snapshot(self, **_kwargs):
        yield self.source


@pytest.mark.parametrize(
    ("constructor", "kind", "operations"),
    [
        (
            "summary",
            "unit_summary",
            {"export_moments", "unit_counts", "readout_snapshot"},
        ),
        ("panel", "unit_panel", {"export_moments", "unit_counts", "readout_snapshot"}),
    ],
)
def test_public_frame_constructors_declare_evidence(constructor, kind, operations) -> None:
    if constructor == "summary":
        analysis = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    else:
        analysis = Analysis.from_unit_panel(
            unit_panel_table(),
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
        )

    source = cast("Any", held_source(analysis))
    evidence = source.evidence
    assert evidence.capture == "snapshot"
    assert evidence.operations == operations
    assert pickle.loads(pickle.dumps(evidence)) == evidence
    assert copy.deepcopy(evidence) == evidence
    analysis.close()


def test_switchback_declares_no_arm_moment_grains() -> None:
    analysis = switchback_analysis()
    source = cast("Any", held_source(analysis))

    evidence = source.evidence
    assert evidence.kind == "switchback"
    assert evidence.grains == frozenset()
    assert evidence.breakouts == ()
    assert evidence.shape is None
    assert evidence.operations == frozenset()
    assert evidence.unit_rows is False
    assert evidence.raw_outcomes is False
    assert evidence.method_fit == "none"
    assert evidence.compliance_axis == "none"
    assert evidence.count_grains == frozenset()
    assert evidence.populations == frozenset({"assigned"})
    assert evidence.sequential_capture == "none"
    assert evidence.scalar is None
    assert evidence.available_covariates is None
    assert source.contrast_stats(source.metrics[0]) is not None
    analysis.close()


@pytest.mark.parametrize(
    ("constructor", "expected"),
    [
        (
            "summary",
            {
                "kind": "unit_summary",
                "capture": "snapshot",
                "grains": frozenset({"total"}),
                "breakouts": (),
                "shape": "unit_summary",
                "operations": frozenset({"export_moments", "unit_counts", "readout_snapshot"}),
                "unit_rows": True,
                "raw_outcomes": True,
                "method_fit": "local",
                "compliance_axis": "none",
                "count_grains": frozenset({"unit"}),
                "populations": frozenset({"assigned"}),
                "sequential_capture": "none",
                "scalar_covariate_grains": None,
                "available_covariates": frozenset({"user_id", "segment", "variant", "revenue"}),
            },
        ),
        (
            "panel",
            {
                "kind": "unit_panel",
                "capture": "snapshot",
                "grains": frozenset({"total", "daily", "asof"}),
                "breakouts": ("segment",),
                "shape": "unit_panel",
                "operations": frozenset({"export_moments", "unit_counts", "readout_snapshot"}),
                "unit_rows": True,
                "raw_outcomes": False,
                "method_fit": "local",
                "compliance_axis": "asof",
                "count_grains": frozenset({"unit"}),
                "populations": frozenset({"assigned"}),
                "sequential_capture": "asof",
                "scalar_covariate_grains": frozenset({"total"}),
                "available_covariates": frozenset(
                    {"user_id", "segment", "ds", "variant", "revenue"}
                ),
            },
        ),
    ],
)
def test_frame_evidence_matches_baseline(constructor, expected) -> None:
    if constructor == "summary":
        analysis = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=(),
        )
    else:
        analysis = Analysis.from_unit_panel(
            unit_panel_table(),
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=("segment",),
        )
    source = cast("Any", held_source(analysis))
    evidence = source.evidence

    assert evidence.kind == expected["kind"]
    assert evidence.grains == expected["grains"]
    assert evidence.breakouts == expected["breakouts"]
    assert evidence.shape == expected["shape"]
    assert evidence.operations == expected["operations"]
    assert evidence.capture == expected["capture"]
    assert evidence.unit_rows == expected["unit_rows"]
    assert evidence.raw_outcomes == expected["raw_outcomes"]
    assert evidence.method_fit == expected["method_fit"]
    assert evidence.compliance_axis == expected["compliance_axis"]
    assert evidence.count_grains == expected["count_grains"]
    assert evidence.populations == expected["populations"]
    assert evidence.sequential_capture == expected["sequential_capture"]
    assert evidence.available_covariates == expected["available_covariates"]
    scalar = evidence.scalar
    assert scalar is not None
    assert scalar.bindings is None
    assert scalar.moments_only is False
    assert scalar.recipe_fixed is False
    assert scalar.recipe_configs == ()
    assert scalar.anonymous_metrics == frozenset()
    assert scalar.method_evidence is True
    assert scalar.breakout_evidence is True
    assert scalar.covariate_grains == expected["scalar_covariate_grains"]
    analysis.close()


def test_declared_panel_grains_match_moment_behavior() -> None:
    analysis = Analysis.from_unit_panel(
        unit_panel_table(),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=("segment",),
    )
    source = cast("Any", held_source(analysis))
    metric = source.context.metrics[0]

    for grain in source.evidence.grains:
        rows = request_evidence(
            evidence_provider(source), MomentsRequest(metric, grain=grain, by=("segment",))
        )
        assert isinstance(cast(MomentRows, rows), tuple)
    analysis.close()


def test_restricted_summary_breakout_declaration_matches_baseline() -> None:
    analysis = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=["segment"],
    )
    source = cast("Any", held_source(analysis))
    breakout = source.summary_breakout_source
    assert breakout is not None
    evidence = breakout.evidence
    assert evidence.kind == "breakout"
    assert evidence.grains == frozenset({"total"})
    assert evidence.breakouts == ("segment",)
    assert evidence.shape == "unit_panel"
    assert evidence.operations == frozenset({"export_moments", "unit_counts", "readout_snapshot"})
    with breakout.readout_snapshot(metrics=(), population="assigned") as scoped:
        assert scoped.evidence.kind == "breakout"
    assert evidence.unit_rows is True
    assert evidence.raw_outcomes is False
    assert evidence.method_fit == "local"
    assert evidence.compliance_axis == "asof"
    assert evidence.count_grains == frozenset({"unit"})
    assert evidence.populations == frozenset({"assigned"})
    assert evidence.sequential_capture == "none"
    assert evidence.scalar is not None
    assert evidence.scalar.bindings is None
    assert evidence.scalar.moments_only is False
    assert evidence.scalar.recipe_fixed is False
    assert evidence.scalar.recipe_configs == ()
    assert evidence.scalar.anonymous_metrics == frozenset()
    assert evidence.scalar.method_evidence is False
    assert evidence.scalar.breakout_evidence is True
    assert evidence.scalar.covariate_grains == frozenset({"total"})
    assert evidence.available_covariates == frozenset(
        {"__summary_day__", "revenue", "segment", "user_id", "variant"}
    )
    analysis.close()


@pytest.mark.integration
def test_definitions_and_pinned_sources_match_baseline(seeded_con, seeded_defs) -> None:
    from increment.errors import CapabilityError

    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con, store="none")
    source = cast("Any", held_source(analysis))
    evidence = source.evidence

    assert evidence.kind == "definitions"
    assert evidence.grains == frozenset({"total", "daily", "asof"})
    assert evidence.breakouts == ("country",)
    assert evidence.shape is None
    assert evidence.operations == frozenset(
        {
            "allocation_history",
            "breakout_source",
            "breakout_sources",
            "breakout_summaries",
            "dashboard_group_data",
            "day_source",
            "exploratory_source",
            "export_moments",
            "factor_summaries",
            "materialize",
            "moments_source",
            "panel_sql",
            "readout_execution",
            "readout_snapshot",
            "sitewide_evidence",
            "summary_sql",
            "triggered_counts",
            "triggered_source",
            "unit_counts",
        }
    )
    assert evidence.unit_rows is True
    assert evidence.raw_outcomes is True
    assert evidence.method_fit == "warehouse"
    assert evidence.compliance_axis == "none"
    assert evidence.count_grains == frozenset({"unit", "assignment"})
    assert evidence.populations == frozenset({"assigned"})
    assert evidence.sequential_capture == "asof"
    assert evidence.scalar is not None
    assert evidence.scalar.bindings is None
    assert evidence.scalar.moments_only is False
    assert evidence.scalar.recipe_fixed is False
    assert evidence.scalar.recipe_configs == ()
    assert evidence.scalar.anonymous_metrics == frozenset()
    assert evidence.scalar.method_evidence is True
    assert evidence.scalar.breakout_evidence is True
    assert evidence.scalar.covariate_grains is None
    assert evidence.available_covariates is None

    metric = source.context.metrics[0]
    with source.readout_snapshot(metrics=(metric,), population="assigned") as pinned:
        assert pinned.evidence == evidence
    with pytest.raises(CapabilityError):
        source.triggered_source()
    analysis.close()


@pytest.mark.integration
def test_artifact_facade_evidence_matches_retained_methods(tmp_path) -> None:
    analysis = unit_day_artifact_analysis(tmp_path)
    source = cast("Any", held_source(analysis))
    evidence = source.evidence
    metric = source.context.metrics[0]

    assert evidence.unit_rows is True
    assert evidence.raw_outcomes is True
    assert evidence.method_fit == "warehouse"
    assert evidence.count_grains == frozenset({"unit", "assignment", "triggered"})
    assert request_evidence(evidence_provider(source), UnitRowsRequest(metric)) is not None
    assert (
        request_evidence(evidence_provider(source), UnitRowsRequest(metric, outcome_stage="raw"))
        is not None
    )
    assert request_evidence(evidence_provider(source), CountsRequest(kind="unit"))
    assert request_evidence(evidence_provider(source), CountsRequest(kind="assignment"))
    assert request_evidence(evidence_provider(source), CountsRequest(kind="triggered"))
    triggered = source.triggered_source()
    assert triggered.evidence.populations == frozenset({"triggered"})
    analysis.close()


@pytest.mark.integration
def test_artifact_without_trigger_extension_declares_assigned_only(tmp_path) -> None:
    from tests.test_unit_day_artifact_facade import _published

    connection, native, context, store, reference = _published()
    analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    source = cast("Any", held_source(analysis))

    assert source.evidence.populations == frozenset({"assigned"})
    assert "triggered_source" not in source.evidence.operations
    analysis.close()
    native.close()
    connection.disconnect()


@pytest.mark.integration
def test_triggered_wrapper_capabilities_match_its_methods(tmp_path) -> None:
    from increment.errors import CapabilityError
    from tests.source_conformance import definitions_source

    source, _ = definitions_source(tmp_path, trigger=True)
    triggered = source.triggered_source()
    metric = triggered.context.metrics[0]
    evidence = triggered.evidence

    assert evidence.unit_rows is True
    assert evidence.raw_outcomes is True
    assert evidence.method_fit == "warehouse"
    assert evidence.count_grains == frozenset({"unit"})
    assert evidence.populations == frozenset({"triggered"})
    assert evidence.sequential_capture == "none"
    assert evidence.operations == frozenset({"readout_execution", "readout_snapshot"})
    assert evidence.scalar is not None
    assert evidence.scalar.method_evidence is True
    assert evidence.scalar.breakout_evidence is False
    assert (
        request_evidence(
            evidence_provider(triggered),
            UnitRowsRequest(metric, population="triggered"),
        )
        is not None
    )
    assert (
        request_evidence(
            evidence_provider(triggered),
            UnitRowsRequest(metric, outcome_stage="raw", population="triggered"),
        )
        is not None
    )
    assert request_evidence(
        evidence_provider(triggered), CountsRequest(kind="unit", population="triggered")
    )
    with pytest.raises(CapabilityError):
        request_evidence(
            evidence_provider(triggered), CountsRequest(kind="cluster", population="triggered")
        )
    source.close()


@pytest.mark.integration
def test_wrappers_refuse_undeclared_grains_before_delegating(tmp_path, monkeypatch) -> None:
    from increment.errors import CapabilityError
    from tests.source_conformance import definitions_source

    source, _ = definitions_source(tmp_path, trigger=True)
    triggered = source.triggered_source()
    metric = triggered.context.metrics[0]
    baseline_rows = cast(
        MomentRows, request_evidence(evidence_provider(source), MomentsRequest(metric))
    )
    original_moments = type(source).moments
    reads = 0

    def counted_moments(self, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original_moments(self, *args, **kwargs)

    monkeypatch.setattr(type(source), "moments", counted_moments)
    try:
        for grain in ("daily", "asof"):
            with pytest.raises(CapabilityError) as raised:
                request_evidence(evidence_provider(triggered), MomentsRequest(metric, grain=grain))
            assert raised.value.code == "evidence.grain_unavailable"
            assert reads == 0

        with pytest.raises(CapabilityError) as raised:
            request_evidence(
                evidence_provider(triggered),
                MomentsRequest(metric, method=object(), by=("not_declared",)),
            )
        assert raised.value.code == "evidence.breakout_unavailable"
        assert reads == 0

        total_rows = cast(
            MomentRows,
            request_evidence(
                evidence_provider(triggered), MomentsRequest(metric, population="triggered")
            ),
        )
        assert total_rows == baseline_rows
        assert reads == 1
        assert triggered.evidence.sql_grains == frozenset()
        with pytest.raises(CapabilityError) as raised:
            triggered.sql(grain="daily")
        assert raised.value.code == "evidence.operation_unavailable"
    finally:
        source.close()

    summary = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=("segment",),
    )
    breakout = cast("Any", held_source(summary)).summary_breakout_source
    assert breakout is not None
    with pytest.raises(CapabilityError) as raised:
        request_evidence(
            evidence_provider(breakout),
            MomentsRequest(breakout.context.metrics[0], grain="daily", by=("segment",)),
        )
    assert raised.value.code == "evidence.grain_unavailable"
    assert request_evidence(
        evidence_provider(breakout),
        MomentsRequest(breakout.context.metrics[0], grain="total", by=("segment",)),
    )
    summary.close()


def test_breakout_moment_source_requires_exactly_its_dimension() -> None:
    from increment.errors import CapabilityError
    from tests.source_conformance import breakout_moments_source

    source = breakout_moments_source()
    with pytest.raises(CapabilityError) as raised:
        request_evidence(
            evidence_provider(source),
            MomentsRequest(source.context.metrics[0]),
        )
    assert raised.value.code == "evidence.breakout_unavailable"
    assert raised.value.context["requested"] == ()
    source.close()


@pytest.mark.integration
def test_triggered_cluster_declaration_matches_cluster_counts(tmp_path) -> None:
    analysis = _triggered_analysis(tmp_path, cluster=True)
    source = cast("Any", held_source(analysis))
    triggered = source.triggered_source()

    assert triggered.evidence.count_grains == frozenset({"unit", "cluster"})
    assert cast(
        CountEvidence,
        request_evidence(
            evidence_provider(triggered), CountsRequest(kind="cluster", population="triggered")
        ),
    ).counts == {"control": 1, "treatment": 1}
    analysis.close()
    from tests.source_conformance import definitions_source

    source, _ = definitions_source(tmp_path, trigger=True)
    triggered = source.triggered_source()
    metric = triggered.context.metrics[0]
    evidence = triggered.evidence

    assert evidence.unit_rows is True
    assert evidence.raw_outcomes is True
    assert evidence.method_fit == "warehouse"
    assert evidence.count_grains == frozenset({"unit"})
    assert evidence.sequential_capture == "none"
    assert evidence.operations == frozenset({"readout_execution", "readout_snapshot"})
    assert evidence.scalar is not None
    assert evidence.scalar.method_evidence is True
    assert evidence.scalar.breakout_evidence is False
    assert (
        request_evidence(
            evidence_provider(triggered), UnitRowsRequest(metric, population="triggered")
        )
        is not None
    )
    assert (
        request_evidence(
            evidence_provider(triggered),
            UnitRowsRequest(metric, outcome_stage="raw", population="triggered"),
        )
        is not None
    )
    assert request_evidence(
        evidence_provider(triggered), CountsRequest(kind="unit", population="triggered")
    )
    source.close()


@pytest.mark.parametrize("constructor", ["summary", "panel", "breakout"])
def test_call_time_covariate_preflight_keeps_its_code(constructor) -> None:
    import pyarrow as pa

    from increment.estimation.engine import Method
    from increment.estimators import InvalidRequestError

    if constructor == "summary":
        frame = unit_summary_table().append_column("pre", pa.array(["not numeric"] * 8))
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=("segment",),
        )
    else:
        table = unit_panel_table()
        frame = table.append_column("pre", pa.array(["not numeric"] * table.num_rows))
        analysis = Analysis.from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=("segment",),
        )

    method = Method.cuped(covariates="pre")
    with pytest.raises(InvalidRequestError) as raised:
        if constructor == "panel":
            analysis.run_daily_lift(decision_method=method)
        elif constructor == "breakout":
            analysis.run_breakout(sensitivity_methods=(method,))
        else:
            analysis.run(decision_method=method)
    assert raised.value.code == "frame.validation.covariate_not_numeric"
    analysis.close()


def test_undeclared_custom_source_refuses_with_the_shared_code() -> None:
    from increment.estimators import CapabilityError
    from increment.readouts import daily, run

    analysis = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = cast("Any", held_source(analysis))

    legacy = cast("Any", _ForwardingProvider(source))
    try:
        assert run(legacy) == analysis.run()
        assert legacy.scope_calls == 1
        with pytest.raises(CapabilityError) as raised:
            daily(legacy)
        assert raised.value.code == "evidence.grain_unavailable"
        assert set(raised.value.context) == {"constructor", "requested", "offered", "route"}
    finally:
        analysis.close()


def test_undeclared_custom_source_keeps_raw_outcomes_and_unit_rows() -> None:
    from increment.readouts import run

    analysis = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[{"name": "revenue", "winsorization": {"upper_percentile": 0.75}}],
    )
    source = cast("Any", held_source(analysis))

    legacy = cast("Any", _ForwardingProvider(source))
    try:
        baseline = analysis.run()
        estimate = next(iter(baseline)).model_dump()
        assert estimate["abs_diff"] == pytest.approx(2.375)
        assert estimate["abs_lb"] == pytest.approx(0.21680716886679186)
        assert estimate["abs_ub"] == pytest.approx(4.726397841615909)
        assert estimate["winsor_upper_bound"] == pytest.approx(5.25)
        assert run(legacy) == baseline
        assert legacy.scope_calls == 1
        metric = analysis.metrics[0]
        assert (
            cast(
                UnitRows,
                request_evidence(evidence_provider(legacy), UnitRowsRequest(metric)),
            ).to_pylist()
            == cast(
                UnitRows,
                request_evidence(evidence_provider(source), UnitRowsRequest(metric)),
            ).to_pylist()
        )
    finally:
        analysis.close()


def test_scalar_covariate_grain_declaration_matches_day_axis_behavior() -> None:
    import pyarrow as pa

    from increment.estimation.engine import Method
    from increment.estimators import InvalidRequestError

    rows = unit_panel_table().to_pylist()
    frame = pa.Table.from_pylist([{**row, "pre": float(int(row["user_id"][1:]))} for row in rows])
    analysis = Analysis.from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = cast("Any", held_source(analysis))
    evidence = source.evidence
    assert evidence.scalar is not None
    assert evidence.scalar.covariate_grains == frozenset({"total"})
    method = Method.cuped(covariates="pre")

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(decision_method=method)
    assert raised.value.code == "estimation.cuped.arm_no_covariate"
    assert analysis.run(decision_method=method)
    analysis.close()


def test_moments_loader_declaration_matches_baseline(tmp_path) -> None:
    import pyarrow.parquet as pq

    from increment.errors import CapabilityError

    producer = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    path = tmp_path / "moments.parquet"
    producer.export(path)
    rows = pq.read_table(path).to_pylist()
    producer.close()

    analysis = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    source = cast("Any", held_source(analysis))
    evidence = source.evidence
    assert evidence.kind == "moments"
    assert evidence.grains == frozenset({"total"})
    assert evidence.breakouts == ()
    assert evidence.shape is None
    assert evidence.operations == frozenset({"unit_counts"})
    assert evidence.unit_rows is False
    assert evidence.raw_outcomes is False
    assert evidence.method_fit == "none"
    assert evidence.compliance_axis == "none"
    assert evidence.count_grains == frozenset({"unit"})
    assert evidence.populations == frozenset({"assigned"})
    assert evidence.sequential_capture == "none"
    assert evidence.scalar is not None
    assert evidence.scalar.bindings is None
    assert evidence.scalar.moments_only is True
    assert evidence.scalar.recipe_fixed is False
    assert evidence.scalar.recipe_configs == ()
    assert evidence.scalar.anonymous_metrics == frozenset({"revenue"})
    assert evidence.scalar.method_evidence is True
    assert evidence.scalar.breakout_evidence is False
    assert evidence.scalar.covariate_grains is None
    assert evidence.available_covariates is None
    with pytest.raises(CapabilityError) as raised:
        request_evidence(evidence_provider(source), UnitRowsRequest(source.context.metrics[0]))
    assert getattr(raised.value, "code", None) == "evidence.unit_rows_unavailable"
    analysis.close()


@pytest.mark.parametrize("constructor", ["summary", "panel"])
def test_frame_row_and_raw_outcome_capabilities_match_methods(constructor) -> None:
    import pyarrow as pa

    from increment.errors import CapabilityError

    summary_frame = unit_summary_table().append_column("pre", pa.array([1.0] * 8))
    panel_table = unit_panel_table()
    panel_frame = panel_table.append_column("pre", pa.array([1.0] * panel_table.num_rows))
    if constructor == "summary":
        analysis = Analysis.from_unit_summary(
            summary_frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        raw_supported = True
    else:
        analysis = Analysis.from_unit_panel(
            panel_frame,
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
        )
        raw_supported = False
    source = cast("Any", held_source(analysis))
    metric = source.context.metrics[0]
    evidence = source.evidence
    assert evidence.unit_rows is True
    assert request_evidence(evidence_provider(source), UnitRowsRequest(metric)) is not None
    assert evidence.raw_outcomes is raw_supported
    if raw_supported:
        assert (
            request_evidence(
                evidence_provider(source), UnitRowsRequest(metric, outcome_stage="raw")
            )
            is not None
        )
    else:
        with pytest.raises(CapabilityError) as raised:
            request_evidence(
                evidence_provider(source), UnitRowsRequest(metric, outcome_stage="raw")
            )
        assert raised.value.code == "evidence.raw_outcomes_unavailable"
    analysis.close()


def test_frame_counts_and_compliance_axes_match_methods() -> None:
    from increment.errors import CapabilityError

    summary = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    summary_source = cast("Any", held_source(summary))
    assert summary_source.evidence.count_grains == frozenset({"unit"})
    assert request_evidence(evidence_provider(summary_source), CountsRequest(kind="unit"))
    with pytest.raises(CapabilityError) as summary_cluster:
        summary_source.cluster_counts()
    assert summary_cluster.value.code == "evidence.cluster_unavailable"
    with pytest.raises(CapabilityError) as summary_dates:
        summary_source.compliance_dates()
    assert summary_dates.value.code == "source.frame.compliance_asof"
    summary.close()

    panel = Analysis.from_unit_panel(
        unit_panel_table(),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics={"revenue": "mean"},
    )
    panel_source = cast("Any", held_source(panel))
    assert panel_source.evidence.count_grains == frozenset({"unit"})
    assert request_evidence(evidence_provider(panel_source), CountsRequest(kind="unit"))
    with pytest.raises(CapabilityError) as panel_cluster:
        panel_source.cluster_counts()
    assert panel_cluster.value.code == "evidence.cluster_unavailable"
    assert panel_source.evidence.compliance_axis == "asof"
    assert panel_source.compliance_dates()
    panel.close()


@pytest.mark.parametrize("constructor", ["summary", "panel"])
def test_local_method_fit_uses_declared_unit_rows(constructor) -> None:
    import pyarrow as pa

    from increment.estimation.engine import Method

    rows = unit_summary_table().to_pylist()
    if constructor == "summary":
        frame = pa.Table.from_pylist(
            [{**row, "pre": float(index)} for index, row in enumerate(rows)]
        )
        analysis = Analysis.from_unit_summary(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
    else:
        rows = unit_panel_table().to_pylist()
        frame = pa.Table.from_pylist(
            [{**row, "pre": float(int(row["user_id"][1:]))} for row in rows]
        )
        analysis = Analysis.from_unit_panel(
            frame,
            unit="user_id",
            group="variant",
            date="ds",
            control="control",
            metrics={"revenue": "mean"},
        )
    source = cast("Any", held_source(analysis))
    assert source.evidence.method_fit == "local"
    assert analysis.run(decision_method=Method.cuped(covariates="pre"))
    analysis.close()


@pytest.mark.integration
def test_sql_source_declares_only_methods_it_serves() -> None:
    from increment.errors import CapabilityError
    from tests.source_conformance import sql_totals_source

    source = sql_totals_source()
    evidence = source.evidence
    metric = source.context.metrics[0]
    assert evidence.unit_rows is False
    assert evidence.raw_outcomes is False
    assert request_evidence(evidence_provider(source), CountsRequest(kind="unit"))
    assert evidence.count_grains == frozenset({"unit"})
    with pytest.raises(CapabilityError) as rows:
        request_evidence(evidence_provider(source), UnitRowsRequest(metric))
    assert rows.value.code == "evidence.unit_rows_unavailable"
    with pytest.raises(CapabilityError) as raw:
        request_evidence(evidence_provider(source), UnitRowsRequest(metric, outcome_stage="raw"))
    assert raw.value.code == "evidence.raw_outcomes_unavailable"
    assert set(raw.value.context) == {"constructor", "requested", "offered", "route"}
    source.close()


def test_breakout_moments_declares_no_unit_or_count_rows() -> None:
    from increment.errors import CapabilityError
    from tests.source_conformance import breakout_moments_source

    source = breakout_moments_source()
    metric = source.context.metrics[0]
    evidence = source.evidence
    assert evidence.unit_rows is False
    assert evidence.raw_outcomes is False
    assert evidence.count_grains == frozenset()
    with pytest.raises(CapabilityError) as rows:
        request_evidence(evidence_provider(source), UnitRowsRequest(metric))
    assert rows.value.code == "evidence.unit_rows_unavailable"
    with pytest.raises(CapabilityError) as counts:
        source.unit_counts()
    assert counts.value.code == "source.breakout.assignment_counts"
    source.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "constructor",
    [
        "summary",
        "panel",
        "restricted_breakout",
        "moments",
        "definitions",
        "pinned",
        "triggered",
        "artifact",
        "sql_totals",
        "breakout_moments",
        "switchback",
    ],
)
def test_declared_capabilities_match_public_method_behavior(
    constructor, tmp_path, seeded_con, seeded_defs
) -> None:
    from increment.errors import CodedError

    if constructor == "summary":
        analysis = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        source = cast("Any", held_source(analysis))
    elif constructor == "panel":
        analysis = _encouragement_panel_analysis()
        source = cast("Any", held_source(analysis))
    elif constructor == "restricted_breakout":
        analysis = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
            breakouts=("segment",),
        )
        source = cast("Any", held_source(analysis)).summary_breakout_source
    elif constructor == "moments":
        import pyarrow.parquet as pq

        producer = Analysis.from_unit_summary(
            unit_summary_table(),
            unit="user_id",
            group="variant",
            control="control",
            metrics={"revenue": "mean"},
        )
        path = tmp_path / "capability-moments.parquet"
        producer.export(path)
        moments = pq.read_table(path).to_pylist()
        producer.close()
        analysis = Analysis.from_moments(moments, metrics={"revenue": "mean"}, control="control")
        source = cast("Any", held_source(analysis))
        assert source.evidence.compliance_axis == "none"
    elif constructor in {"definitions", "pinned"}:
        analysis = Analysis.from_definitions(
            "new_onboarding_v2", seeded_defs, seeded_con, store="none"
        )
        source = cast("Any", held_source(analysis))
        if constructor == "pinned":
            metric = source.context.metrics[0]
            with source.readout_snapshot(metrics=(metric,), population="assigned") as pinned:
                _assert_capability_methods_match_behavior(pinned, analysis, CodedError)
            analysis.close()
            return
    elif constructor == "triggered":
        analysis = _triggered_analysis(tmp_path)
        source = cast("Any", held_source(analysis)).triggered_source()
    elif constructor == "artifact":
        analysis = unit_day_artifact_analysis(tmp_path)
        source = cast("Any", held_source(analysis))
    elif constructor in {"sql_totals", "breakout_moments"}:
        from tests.source_conformance import breakout_moments_source, sql_totals_source

        analysis = None
        source = sql_totals_source() if constructor == "sql_totals" else breakout_moments_source()
    else:
        analysis = switchback_analysis()
        source = cast("Any", held_source(analysis))
        evidence = source.evidence
        assert evidence.grains == frozenset()
        assert evidence.unit_rows is False
        assert evidence.raw_outcomes is False
        assert evidence.count_grains == frozenset()
        assert evidence.compliance_axis == "none"
        assert evidence.sequential_capture == "none"
        assert not callable(getattr(source, "moments", None))
        with pytest.raises(CodedError):
            analysis.capture_sequential(finalized=True)
        analysis.close()
        return

    _assert_capability_methods_match_behavior(source, analysis, CodedError)
    if analysis is None:
        source.close()
    else:
        analysis.close()


def _assert_capability_methods_match_behavior(source, analysis, error_type) -> None:
    evidence = source.evidence
    metric = source.context.metrics[0]
    by = tuple(evidence.breakouts) if evidence.kind == "breakout" else ()
    population = "triggered" if "triggered" in evidence.populations else "assigned"

    for grain in ("total", "daily", "asof"):
        request = MomentsRequest(metric, grain=grain, by=by, population=population)
        if grain in evidence.grains:
            assert request_evidence(evidence_provider(source), request) is not None
        else:
            with pytest.raises(error_type):
                request_evidence(evidence_provider(source), request)

    if evidence.unit_rows:
        assert (
            request_evidence(
                evidence_provider(source), UnitRowsRequest(metric, population=population)
            )
            is not None
        )
    else:
        with pytest.raises(error_type):
            request_evidence(
                evidence_provider(source), UnitRowsRequest(metric, population=population)
            )
    if evidence.raw_outcomes:
        assert (
            request_evidence(
                evidence_provider(source),
                UnitRowsRequest(metric, outcome_stage="raw", population=population),
            )
            is not None
        )
    else:
        with pytest.raises(error_type):
            request_evidence(
                evidence_provider(source),
                UnitRowsRequest(metric, outcome_stage="raw", population=population),
            )

    count_methods = {
        "unit": "unit_counts",
        "cluster": "cluster_counts",
        "assignment": "assignment_counts",
        "triggered": "triggered_counts",
    }
    for grain, method_name in count_methods.items():
        method = getattr(source, method_name, None)
        if not callable(method):
            assert grain not in evidence.count_grains
        elif grain in evidence.count_grains:
            request = CountsRequest(kind=cast("Any", grain), population=population)
            assert request_evidence(evidence_provider(source), request) is not None
        else:
            with pytest.raises(error_type):
                request_evidence(
                    evidence_provider(source),
                    CountsRequest(kind=cast("Any", grain), population=population),
                )

    compliance_dates = getattr(source, "compliance_dates", None)
    if evidence.compliance_axis == "asof":
        assert callable(compliance_dates)
        assert compliance_dates() is not None
    elif callable(compliance_dates):
        with pytest.raises((error_type, AttributeError)):
            compliance_dates()

    capture = getattr(source, "capture_sequential", None)
    if evidence.sequential_capture == "none":
        if analysis is not None:
            with pytest.raises(error_type):
                analysis.capture_sequential(finalized=True)
        elif callable(capture):
            with pytest.raises(error_type):
                capture(finalized=True)
        else:
            sequential_snapshot = getattr(source, "sequential_snapshot", None)
            if callable(sequential_snapshot):
                with pytest.raises(error_type):
                    sequential_snapshot()
    else:
        assert callable(capture)


def _triggered_analysis(tmp_path, *, cluster=False):
    import datetime as dt

    import ibis
    import yaml

    from increment.artifacts import SourceSnapshotEvidence

    events = []
    for index, group in enumerate(("control", "control", "treatment", "treatment")):
        user = f"triggered-{index}"
        events.extend(
            [
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 9),
                    "event": "assignment",
                    "group_id": group,
                    "revenue": None,
                },
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 10),
                    "event": "trigger",
                    "group_id": group,
                    "revenue": None,
                },
                {
                    "user_id": user,
                    "ts": dt.datetime(2026, 1, 1, 12),
                    "event": "purchase",
                    "group_id": None,
                    "revenue": float(index + 1),
                },
            ]
        )
    if cluster:
        for event in events:
            event["store_id"] = f"store-{event['group_id'] or 'outcome'}"
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
                "name": "triggered",
                "exposure": "assignment",
                "trigger": "trigger",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "control_group": "control",
                **({"cluster": "store_id"} if cluster else {}),
                "plan": {"primary": "revenue"},
            }
        ],
    }
    definitions_path = tmp_path / "triggered-capabilities.yaml"
    definitions_path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    connection = ibis.duckdb.connect()
    connection.create_table("raw_events", obj=events)
    cutoff = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    return Analysis.from_definitions(
        "triggered",
        definitions_path,
        connection,
        store="none",
        source_snapshot_evidence=SourceSnapshotEvidence(cutoff, {"events": cutoff}),
    )


def _registered_unit_summary_analysis():
    import datetime as dt

    import pyarrow as pa

    from increment import AnalysisPlan, InferenceSpec, MetricSpec
    from tests.asymptotic_cases import mean_model, mean_registration
    from tests.test_sequential_public_sources import gaussian_plan

    rows = unit_summary_table().to_pylist()
    for index, row in enumerate(rows):
        row["exposure"] = dt.date(2026, 1, 1) + dt.timedelta(days=index)
    frame = pa.Table.from_pylist(rows)
    specs = [MetricSpec(name="revenue", type="mean")]
    template = gaussian_plan(
        specs,
        source_id="frame",
        unit="user_id",
        group="variant",
        exposure_date="exposure",
    )
    template_registration = template.inference.registration
    assert template_registration is not None
    registration = mean_registration(
        models=(mean_model("revenue"),),
        cells=template_registration.roster,
        source_id="frame",
        definitions_id=template_registration.definitions_id,
    )
    plan = AnalysisPlan(
        primary="revenue",
        inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
    )
    return Analysis.from_unit_summary(
        frame,
        unit="user_id",
        group="variant",
        control="control",
        metrics=specs,
        experiment_id="frame",
        plan=plan,
        exposure_date="exposure",
    )


def test_registered_unit_summary_declares_construction_capture() -> None:
    analysis = _registered_unit_summary_analysis()
    source = cast("Any", held_source(analysis))

    assert source.evidence.sequential_capture == "construction"
    assert analysis.capture_sequential(finalized=True).records
    analysis.close()


def test_moments_with_restored_checkpoint_declares_replay_capture(tmp_path) -> None:
    import pyarrow.parquet as pq

    from increment import MetricSpec

    analysis = _registered_unit_summary_analysis()
    expected = analysis.sequential_snapshot()
    path = tmp_path / "registered-checkpoint.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    analysis.close()

    replay = Analysis.from_moments(
        payload,
        metrics=[MetricSpec(name="revenue", type="mean")],
        control="control",
    )
    source = cast("Any", held_source(replay))

    assert source.evidence.sequential_capture == "replay"
    assert replay.sequential_snapshot() == expected
    replay.close()


def _encouragement_summary_analysis():
    import pyarrow as pa

    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    rows = unit_summary_table().to_pylist()
    for row in rows:
        row["clicked"] = int(row["variant"] == "treatment")
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment changes revenue through uptake"
        ),
    )
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="user_id",
        group="variant",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
    )
    return analysis, design


def _encouragement_panel_analysis():
    import pyarrow as pa

    from increment import Encouragement, ExclusionRestriction, UptakeSpec

    rows = unit_panel_table().to_pylist()
    frame = pa.Table.from_pylist(
        [{**row, "clicked": int(row["variant"] == "treatment")} for row in rows]
    )
    design = Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            justification="assignment changes revenue through uptake"
        ),
    )
    return Analysis.from_unit_panel(
        frame,
        unit="user_id",
        group="variant",
        date="ds",
        metrics={"revenue": "mean"},
        uptake="clicked",
        design=design,
        breakouts=("segment",),
    )


def test_summary_and_moments_declare_retained_total_compliance(tmp_path) -> None:
    import pyarrow.parquet as pq

    analysis, design = _encouragement_summary_analysis()
    source = cast("Any", held_source(analysis))
    summary = cast(
        ComplianceSummary,
        request_evidence(evidence_provider(source), ComplianceRequest(design)),
    )
    assert summary.as_of is None
    assert source.evidence.compliance_axis == "total"

    path = tmp_path / "encouragement-summary.parquet"
    analysis.export(path)
    payload = pq.read_table(path).to_pylist()
    analysis.close()

    replay = Analysis.from_moments(payload, metrics={"revenue": "mean"}, design=design)
    replay_source = cast("Any", held_source(replay))
    assert replay_source.evidence.compliance_axis == "total"
    assert (
        cast(
            ComplianceSummary,
            request_evidence(evidence_provider(replay_source), ComplianceRequest(design)),
        )
        == summary
    )
    replay.close()


@pytest.mark.integration
def test_definitions_day_source_declares_only_its_day_axis(seeded_con, seeded_defs) -> None:
    from increment.errors import CapabilityError

    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con, store="none")
    source = cast("Any", held_source(analysis))
    day = source.day_source(metrics=(source.context.metrics[0],))

    assert day.evidence.grains == frozenset({"daily", "asof"})
    assert day.evidence.unit_rows is False
    assert day.evidence.method_fit == "none"
    assert day.evidence.compliance_axis == "none"
    assert day.evidence.count_grains == frozenset()
    assert day.evidence.populations == frozenset({"assigned"})
    with pytest.raises(CapabilityError):
        request_evidence(evidence_provider(day), UnitRowsRequest(source.context.metrics[0]))
    with pytest.raises(CapabilityError):
        request_evidence(evidence_provider(day), CountsRequest(kind="unit"))
    if source.context.trigger_name is not None:
        triggered = source.day_source(metrics=(source.context.metrics[0],), population="triggered")
        assert triggered.evidence.populations == frozenset({"triggered"})
    analysis.close()


@pytest.mark.filterwarnings("ignore:metric .* has window_days=None")
@pytest.mark.integration
def test_artifact_count_declarations_follow_retained_extensions(tmp_path) -> None:
    import datetime as dt

    import ibis

    from increment.artifacts import SourceSnapshotEvidence
    from increment.errors import InvalidRequestError
    from increment.query.session import WarehouseArtifactStore
    from tests.test_analysis_trigger import (
        _clustered_trigger_events,
        _defs_yaml,
        _defs_yaml_clustered,
        _events,
    )
    from tests.test_unit_day_artifact_facade import _expected_context, _extensions, _published

    definitions = tmp_path / "trigger.yaml"
    definitions.write_text(_defs_yaml())
    connection = ibis.duckdb.connect()
    connection.create_table("events", obj=_events(n_per_arm=30))
    native = Analysis.from_definitions(
        "exp",
        definitions,
        connection,
        source_snapshot_evidence=SourceSnapshotEvidence(dt.datetime(2024, 1, 31, tzinfo=dt.UTC)),
    )
    artifact_context = _expected_context(definitions, "exp")
    store = WarehouseArtifactStore(connection, schema_name="capabilities")
    reference = native.publish_unit_day_artifact(
        store,
        extensions=_extensions(artifact_context, "trigger_population"),
    )
    analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=artifact_context)
    source = cast("Any", held_source(analysis))
    triggered = source.triggered_source()

    assert "unit" not in triggered.evidence.count_grains
    with pytest.raises(InvalidRequestError) as missing_counts:
        triggered.unit_counts()
    assert missing_counts.value.code == "artifact.extension.missing"
    analysis.close()
    native.close()
    connection.disconnect()

    cluster_definitions = tmp_path / "cluster.yaml"
    cluster_definitions.write_text(
        _defs_yaml_clustered(trigger=None)
        .replace("    window_days: 7\n", "")
        .replace(
            "start: 2024-01-01T00:00:00",
            "start: 2024-01-01T00:00:00\n    end: 2024-01-09T00:00:00",
        )
    )
    connection = ibis.duckdb.connect()
    connection.create_table(
        "cluster_trigger_events", obj=_clustered_trigger_events(n_stores_per_arm=4)
    )
    native = Analysis.from_definitions(
        "exp",
        cluster_definitions,
        connection,
        source_snapshot_evidence=SourceSnapshotEvidence(dt.datetime(2024, 1, 9, tzinfo=dt.UTC)),
    )
    artifact_context = _expected_context(cluster_definitions, "exp")
    store = WarehouseArtifactStore(connection, schema_name="cluster_capabilities")
    reference = native.publish_unit_day_artifact(store)
    analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=artifact_context)
    source = cast("Any", held_source(analysis))

    assert "assignment" not in source.evidence.count_grains
    assert "cluster" not in source.evidence.count_grains
    assert request_evidence(evidence_provider(source), CountsRequest(kind="unit"))
    with pytest.raises(InvalidRequestError) as missing_assignment_counts:
        request_evidence(evidence_provider(source), CountsRequest(kind="assignment"))
    assert missing_assignment_counts.value.code == "artifact.evidence.unavailable"
    analysis.close()
    native.close()
    connection.disconnect()

    connection, native, context, store, reference = _published()
    analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    source = cast("Any", held_source(analysis))
    assert source.evidence.count_grains == frozenset({"unit", "assignment"})
    assert (
        cast(
            CountEvidence,
            request_evidence(evidence_provider(source), CountsRequest(kind="assignment")),
        ).counts
        == cast(
            CountEvidence, request_evidence(evidence_provider(source), CountsRequest(kind="unit"))
        ).counts
    )
    analysis.close()
    native.close()
    connection.disconnect()


@pytest.mark.integration
def test_artifact_facade_keeps_warehouse_fit_declaration(tmp_path) -> None:
    analysis = unit_day_artifact_analysis(tmp_path)
    source = cast("Any", held_source(analysis))

    assert source.evidence.method_fit == "warehouse"
    assert callable(source.fit_cuped_multicovariate)
    analysis.close()
