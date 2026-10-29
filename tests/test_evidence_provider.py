from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Literal, cast

import pytest

from increment import Analysis, EvidenceCapabilities, Method
from increment.errors import CapabilityError, CodedError
from increment.evidence import (
    BreakoutIdentity,
    BreakoutSlicesRequest,
    BreakoutUnitRowsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountEvidence,
    CountsRequest,
    DayAxisRequest,
    EvidenceProvider,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitAnswer,
    MethodFitRequest,
    MomentsRequest,
    SequentialCaptureRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
    require_evidence,
)
from increment.semantics.design import Encouragement
from tests.analysis_factory import held_source
from tests.source_conformance import (
    definitions_source,
    frame_panel_source,
    frame_summary_source,
    provider_adapters,
    sql_totals_source,
    unit_summary_table,
)


def _frame_rows(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_pylist"):
        return frame.to_pylist()
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    return frame.to_dict(orient="records")


def _source_for_population(source: Any, population: str) -> Any:
    populations = source.evidence.populations
    if population == "triggered" and not (
        "triggered" in populations and "assigned" not in populations
    ):
        return source.triggered_source()
    return source


def _population(value: str) -> Literal["assigned", "triggered"]:
    return cast(Literal["assigned", "triggered"], value)


def _assert_moments_and_unit_rows(
    provider: Any, source: Any, adapter: Any, metric: Any, populations: frozenset[str]
) -> None:
    for population in populations:
        expected_source = _source_for_population(source, population)
        for grain in source.evidence.grains:
            by = source.evidence.breakouts if adapter.name == "breakout_moments" else ()
            request = MomentsRequest(metric, grain=grain, by=by, population=_population(population))
            expected = tuple(
                expected_source.moments(
                    metric,
                    grain=grain,
                    by=by,
                    include_covariate=request.include_covariate,
                    completed_windows_only=request.completed_windows_only,
                )
            )
            assert provider.moments(request) == expected
        if source.evidence.unit_rows:
            stages = ("transformed", "raw") if source.evidence.raw_outcomes else ("transformed",)
            for stage in stages:
                request = UnitRowsRequest(
                    metric, outcome_stage=stage, population=_population(population)
                )
                expected = expected_source.unit_frame(
                    metric, covariates=request.covariates, outcome_stage=stage
                )
                assert _frame_rows(provider.unit_rows(request)) == _frame_rows(expected)


def _assert_counts(provider: Any, source: Any, populations: frozenset[str]) -> None:
    for kind in source.evidence.count_grains:
        population = (
            "triggered"
            if "triggered" in populations and kind in {"unit", "cluster"}
            else "assigned"
        )
        actual = provider.counts(CountsRequest(kind, population=_population(population)))
        if kind == "triggered":
            grain, counts, units = source.triggered_counts()
            expected = CountEvidence("triggered", grain, counts, units)
        elif population == "triggered" and callable(getattr(source, "triggered_counts", None)):
            grain, counts, units = source.triggered_counts()
            expected = CountEvidence(
                population,
                "unit" if kind == "unit" else "cluster",
                (counts if grain == "unit" else units) if kind == "unit" else counts,
                units,
            )
        else:
            expected_source = _source_for_population(source, population)
            if kind == "unit":
                expected = CountEvidence(population, "unit", expected_source.unit_counts())
            elif kind == "cluster":
                expected = CountEvidence(
                    population,
                    "cluster",
                    expected_source.cluster_counts(),
                    expected_source.unit_counts(),
                )
            else:
                expected = CountEvidence(
                    population,
                    "assignment",
                    expected_source.assignment_counts(population=population),
                )
        assert actual == expected


def _assert_method_fit(
    provider: Any, source: Any, metric: Any, populations: frozenset[str]
) -> None:
    if source.evidence.method_fit == "none":
        return
    fit_method = Method.cuped_multicovariate(covariates=("revenue",))
    control = source.context.design.control_group
    arm_order = tuple(sorted(source.unit_counts()))
    population = (
        "triggered" if "triggered" in populations and "assigned" not in populations else "assigned"
    )
    answer = provider.method_fit(
        MethodFitRequest(
            metric=metric,
            method=fit_method,
            control=control,
            arm_order=arm_order,
            population=_population(population),
        )
    )
    fit = getattr(source, "fit_cuped_multicovariate", None)
    if not callable(fit):
        assert answer == MethodFitAnswer("unavailable")
        return
    try:
        identity, envelope = fit(
            metric,
            fit_method,
            control=control,
            arm_order=arm_order,
            population=population,
        )
    except CodedError as error:
        assert answer.failure is not None
        assert (answer.failure.code, answer.failure.context) == (error.code, error.context)
        return
    assert answer.fit_identity == tuple(identity)
    if isinstance(envelope, CodedError):
        assert answer.failure is not None
        assert (answer.failure.code, answer.failure.context) == (envelope.code, envelope.context)
    else:
        assert answer.envelope == envelope


def _assert_compliance(provider: Any, source: Any, populations: frozenset[str]) -> None:
    if source.evidence.compliance_axis == "none" or not isinstance(
        source.context.design, Encouragement
    ):
        return
    design = source.context.design
    assert provider.compliance(ComplianceRequest(design)) == source.compliance_summary(design)
    for population in populations:
        series_source = _source_for_population(source, population)
        request = ComplianceSeriesRequest(design, population=_population(population))
        expected = tuple(
            (day, series_source.compliance_summary(design, as_of=day))
            for day in series_source.compliance_dates()
        )
        assert provider.compliance_series(request) == expected


@pytest.mark.integration
def test_each_builtin_arm_adapter_matches_every_declared_evidence_path(tmp_path) -> None:
    for adapter in provider_adapters(tmp_path):
        source, metric = adapter.build()
        provider = evidence_provider(source)
        populations = source.evidence.populations
        _assert_moments_and_unit_rows(provider, source, adapter, metric, populations)
        _assert_counts(provider, source, populations)
        _assert_method_fit(provider, source, metric, populations)
        _assert_compliance(provider, source, populations)
        for population in populations:
            grain = next(iter(source.evidence.grains), "total")
            scope_request = EvidenceScope(
                metrics=(metric,),
                population=population,
                grain=grain,
                snapshot=True,
                execution=True,
            )
            scope = provider.scope(scope_request)
            assert isinstance(scope, EvidenceScopeResult)
            assert replace(scope.scope, uptake_facts=()) == scope_request
            assert scope.scope.uptake_facts in ((), ("clicked",))
            by = source.evidence.breakouts if adapter.name == "breakout_moments" else ()
            moments_request = MomentsRequest(metric, grain=grain, by=by, population=population)
            expected = provider.moments(moments_request)
            with scope.source as scoped_source:
                scoped_provider = evidence_provider(scoped_source)
                actual = scoped_provider.moments(moments_request)
                assert actual == expected
        if source.evidence.sequential_capture == "none":
            with pytest.raises(CapabilityError) as exc_info:
                provider.capture_sequential(SequentialCaptureRequest(finalized=True))
            assert exc_info.value.code == "evidence.operation_unavailable"
        else:
            assert adapter.analysis_factory is not None, adapter.name
            analysis = adapter.analysis_factory()
            try:
                from datetime import date

                for freeze in ((), (adapter.metric_name,)):
                    request = SequentialCaptureRequest(
                        finalized=True, as_of=date(2026, 1, 3), freeze=freeze
                    )
                    try:
                        expected = analysis.capture_sequential(
                            finalized=True, as_of=date(2026, 1, 3), freeze=freeze
                        )
                    except CapabilityError as expected_error:
                        with pytest.raises(CapabilityError) as actual_error:
                            provider.capture_sequential(request)
                        assert actual_error.value.code == expected_error.code
                    else:
                        actual = provider.capture_sequential(request)
                        assert actual == expected
                        assert actual.states == expected.states
                        assert actual.records == expected.records
                        assert actual.frozen == expected.frozen
            finally:
                analysis.close()
        close = getattr(source, "close", None)
        if callable(close):
            close()


@pytest.mark.integration
def test_compliance_requests_match_the_source_summary_path() -> None:
    from datetime import date

    import pyarrow as pa

    from increment.semantics.design import Encouragement

    design = Encouragement.model_validate(
        {
            "mechanism": "encouragement",
            "control_group": "control",
            "uptake": {"fact": "clicked"},
            "exclusion_restriction": {"justification": "the prompt has no direct effect"},
        }
    )
    rows = [
        {
            "unit": f"{arm}-{index}",
            "arm": arm,
            "day": date(2026, 1, index + 1),
            "exposure": date(2026, 1, 1),
            "revenue": float(index + 1),
            "clicked": int(arm == "treatment" and index % 2 == 0),
        }
        for index in range(2)
        for arm in ("control", "treatment")
    ]
    analysis = Analysis.from_unit_panel(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="arm",
        date="day",
        exposure_date="exposure",
        uptake="clicked",
        observation_end=date(2026, 1, 4),
        design=design,
        metrics={"revenue": "mean"},
    )
    try:
        source = cast(Any, held_source(analysis))
        provider = evidence_provider(source)

        summary = provider.compliance(ComplianceRequest(design))
        assert summary == source.compliance_summary(design)

        actual_series = provider.compliance_series(ComplianceSeriesRequest(design))
        expected_series = tuple(
            (day, source.compliance_summary(design, as_of=day)) for day in source.compliance_dates()
        )
        assert actual_series == expected_series
    finally:
        analysis.close()


@pytest.mark.integration
def test_triggered_moment_request_matches_triggered_source_path(tmp_path) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True)
    metric = source.context.metrics[0]
    provider = evidence_provider(source)
    expected = tuple(source.triggered_source().moments(metric, grain="total"))
    assigned = tuple(source.moments(metric, grain="total"))

    actual = provider.moments(MomentsRequest(metric, grain="total", population="triggered"))

    assert expected != assigned
    assert actual == expected


@pytest.mark.integration
def test_triggered_day_view_moments_use_its_declared_population(tmp_path) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True)
    metric = source.context.metrics[0]
    day_source = source.day_source(metrics=list(source.context.metrics), population="triggered")
    request = MomentsRequest(metric, grain="daily", population="triggered")

    actual = evidence_provider(day_source).moments(request)

    assert actual == tuple(day_source.moments(metric, grain="daily"))
    assert actual


@pytest.mark.integration
def test_method_moments_refuse_unserved_grain_and_options() -> None:
    source = frame_panel_source()
    metric = source.context.metrics[0]
    method = Method.unadjusted()
    provider = evidence_provider(source)
    expected = tuple(source.moments_for_method(metric, method, by=()))
    assert provider.moments(MomentsRequest(metric, method=method)) == expected

    for request in (
        MomentsRequest(metric, grain="daily", method=method),
        MomentsRequest(metric, method=method, include_covariate=True),
        MomentsRequest(metric, method=method, completed_windows_only=True),
    ):
        with pytest.raises(CapabilityError):
            provider.moments(request)


@pytest.mark.integration
def test_sql_panel_unadjusted_method_uses_ordinary_moments() -> None:
    source = sql_totals_source()
    metric = source.context.metrics[0]
    request = MomentsRequest(metric, method=Method.unadjusted())

    expected = tuple(source.moments(metric, grain=request.grain, by=request.by))
    actual = evidence_provider(source).moments(request)

    assert actual == expected


@pytest.mark.integration
def test_compliance_series_does_not_reconvert_triggered_source() -> None:
    from tests.analysis_factory import _native_source
    from tests.test_compliance_artifacts import native_fixture
    from tests.test_compliance_summary import design

    con, analysis, _, _, _ = native_fixture(("full",), triggered=True)
    try:
        native = _native_source(analysis)
        triggered = native.triggered_source()
        uptake_design = design(7)
        request = ComplianceSeriesRequest(uptake_design, population="triggered")
        expected = tuple(
            (day, triggered.compliance_summary(uptake_design, as_of=day))
            for day in triggered.compliance_dates()
        )

        actual = evidence_provider(triggered).compliance_series(request)

        assert actual == expected
    finally:
        con.disconnect()


@pytest.mark.integration
def test_raw_unit_rows_match_the_loader_path(tmp_path) -> None:
    source, _ = definitions_source(tmp_path)
    metric = source.context.metrics[0]
    provider = evidence_provider(source)
    request = UnitRowsRequest(metric, outcome_stage="raw")

    expected = source.unit_frame(metric, outcome_stage="raw")
    actual = provider.unit_rows(request)

    assert actual.to_pylist() == expected.to_pylist()


@pytest.mark.integration
def test_local_fit_source_reports_unavailable_fit_evidence() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]
    request = MethodFitRequest(
        metric=metric,
        method=Method.cuped_multicovariate(covariates=("segment",)),
        control="control",
        arm_order=("control", "treatment"),
        population="assigned",
    )

    answer = evidence_provider(source).method_fit(request)

    assert answer.origin == "unavailable"
    assert answer.envelope is None


@pytest.mark.integration
def test_triggered_unit_counts_select_units_from_clustered_counts(tmp_path) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True, cluster=True)
    expected = source.triggered_counts()[2]

    actual = evidence_provider(source).counts(CountsRequest("unit", population="triggered"))

    assert actual.grain == "unit"
    assert actual.counts == expected


@pytest.mark.integration
def test_unclustered_triggered_unit_counts_select_reported_unit_grain(tmp_path) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True)

    actual = evidence_provider(source).counts(CountsRequest("unit", population="triggered"))

    assert actual == CountEvidence("triggered", "unit", {"control": 2, "treatment": 1}, {})


@pytest.mark.integration
def test_triggered_assignment_counts_keep_assignment_grain(tmp_path) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True)

    actual = evidence_provider(source).counts(CountsRequest("assignment", population="triggered"))

    assert actual == CountEvidence("triggered", "assignment", {"control": 2, "treatment": 1})


@pytest.mark.integration
def test_triggered_assignment_counts_use_triggered_counts_without_assignment_method(
    tmp_path,
) -> None:
    source, _ = definitions_source(tmp_path, trigger=True, partial_trigger=True)
    expected = source.triggered_counts()[1]

    class LegacyCountsSource:
        evidence = source.evidence
        context = source.context

        def moments(self, request):
            return evidence_provider(source).moments(request)

        def unit_rows(self, request):
            return evidence_provider(source).unit_rows(request)

        def method_fit(self, request):
            return evidence_provider(source).method_fit(request)

        def breakout_method_moments(self, request):
            return evidence_provider(source).breakout_method_moments(request)

        def compliance(self, request):
            return evidence_provider(source).compliance(request)

        def compliance_series(self, request):
            return evidence_provider(source).compliance_series(request)

        def counts(self, request):
            if request.kind == "assignment" and request.population == "triggered":
                _grain, counts, _units = source.triggered_counts()
                return CountEvidence("triggered", "assignment", counts)
            return evidence_provider(source).counts(request)

        def capture_sequential(self, request):
            return evidence_provider(source).capture_sequential(request)

        def scope(self, request):
            return evidence_provider(source).scope(request)

    actual = evidence_provider(LegacyCountsSource()).counts(
        CountsRequest("assignment", population="triggered")
    )
    assert actual == CountEvidence("triggered", "assignment", expected)


@pytest.mark.integration
def test_triggered_count_request_validates_triggered_population_on_artifact_view(tmp_path) -> None:
    from tests.analysis_factory import held_source
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    source = cast(Any, held_source(analysis))
    triggered = source.triggered_source()
    try:
        grain, counts, units = source.triggered_counts()
        expected = CountEvidence("triggered", grain, counts, units)
        request = CountsRequest("triggered")

        assert request_evidence(evidence_provider(source), request) == expected
        assert request_evidence(evidence_provider(triggered), request) == expected
    finally:
        analysis.close()


@pytest.mark.integration
def test_categorical_fit_evidence_precedes_available_multicovariate_fit(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import AnalysisPlan
    from increment.frame import MetricSpec
    from increment.semantics.models import CupedCovariate

    method = Method.cuped_multicovariate(
        name="categorical",
        covariates=(CupedCovariate(name="segment", kind="categorical"),),
    )
    spec = MetricSpec(name="revenue", type="mean", decision_method=method)
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(16)],
                "arm": ["control"] * 8 + ["treatment"] * 8,
                "revenue": [float(i % 4) for i in range(8)]
                + [float(i % 4) + 1.0 for i in range(8)],
                "segment": ["a", "a", "b", "b"] * 4,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[spec],
        plan=AnalysisPlan(primary="revenue"),
    )
    cube_path = tmp_path / "categorical.parquet"
    analysis.export(cube_path)
    replay = Analysis.from_moments(
        pq.read_table(cube_path).to_pylist(),
        metrics=[spec],
        control="control",
    )
    source = cast(Any, held_source(replay))
    try:
        metric = source.context.metrics[0]
        assert callable(getattr(source, "fit_cuped_multicovariate", None))
        expected = source.categorical_evidence(metric, method)
        request = MethodFitRequest(
            metric=metric,
            method=method,
            control="control",
            arm_order=("control", "treatment"),
            population="assigned",
        )

        actual = evidence_provider(source).method_fit(request)

        assert expected is not None
        assert actual.origin == "retained_categorical"
        assert actual.cells == expected.cells
        assert actual.evidence_sha256 == expected.payload_sha256
    finally:
        analysis.close()
        replay.close()


@pytest.mark.integration
def test_assignment_count_refusal_from_moments_cube_is_preserved() -> None:
    from increment import AnalysisPlan
    from increment.frame import MetricSpec

    rows = [
        {
            "experiment_id": "without-counts",
            "metric": "revenue",
            "group_id": group,
            "n": 2,
            "successes": None,
            "ref_y": total,
            "cy1": 0.0,
            "cy2": square,
            "ref_x": None,
            "cx1": None,
            "cx2": None,
            "cxy": None,
            "ref_den": None,
            "cden1": None,
            "cden2": None,
            "cyden": None,
            "sum_d": None,
            "cyd": None,
            "cy2d": None,
            "cxd": None,
            "x_role": None,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "moments_format": 10,
        }
        for group, total, square in (("control", 3.0, 5.0), ("treatment", 7.0, 25.0))
    ]
    analysis = Analysis.from_moments(
        rows,
        metrics=[MetricSpec(name="revenue", type="mean")],
        control="control",
        plan=AnalysisPlan(primary="revenue"),
    )
    source = cast(Any, held_source(analysis))
    try:
        with pytest.raises(CodedError) as raised:
            evidence_provider(source).counts(CountsRequest("assignment"))

        assert raised.value.code == "source.moments.assignment_counts"
    finally:
        analysis.close()


@pytest.mark.integration
def test_sequential_capture_applies_freeze_like_analysis() -> None:
    from increment.frame import MetricSpec
    from tests.test_asymptotic_public_sources import _analysis, _frame, _plan

    specs = [MetricSpec(name="outcome", type="mean")]
    frame = _frame([1, 2] * 8, [5, 7] * 8)
    plan = _plan(specs, start=2)
    expected_analysis = _analysis(frame.copy(), specs, plan)
    actual_analysis = _analysis(frame.copy(), specs, plan)
    source = cast(Any, held_source(actual_analysis))
    try:
        for freeze in ((), ("outcome",)):
            request = SequentialCaptureRequest(finalized=True, freeze=freeze)
            expected = expected_analysis.capture_sequential(finalized=True, freeze=freeze)
            actual = evidence_provider(source).capture_sequential(request)
            assert actual == expected
            assert actual.states == expected.states
            assert actual.records == expected.records
            assert actual.prefix_id == expected.prefix_id
            assert actual.frozen == expected.frozen
    finally:
        expected_analysis.close()
        actual_analysis.close()


@pytest.mark.integration
def test_sequential_capture_matches_registered_temporal_snapshot(con) -> None:
    from datetime import date

    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    previous_analysis = registered_native(_analysis_with_asof_events(con, sequential=True))
    actual_analysis = registered_native(_analysis_with_asof_events(con, sequential=True))
    expected_analysis = registered_native(_analysis_with_asof_events(con, sequential=True))
    try:
        first_horizon = date(2025, 7, 1)
        next_horizon = date(2025, 8, 1)
        previous = previous_analysis.capture_sequential(
            finalized=True, as_of=first_horizon, freeze=("revenue",)
        )
        expected = expected_analysis.capture_sequential(
            finalized=True, previous=previous, as_of=next_horizon
        )
        actual = evidence_provider(held_source(actual_analysis)).capture_sequential(
            SequentialCaptureRequest(
                finalized=True,
                previous=previous,
                as_of=next_horizon,
                freeze=(),
            )
        )

        assert actual.registration_id == expected.registration_id
        assert actual.states == expected.states
        assert actual.records == expected.records
        actual.verify_parent(previous)
        assert actual.frozen == previous.frozen
        assert actual.frozen == expected.frozen
        assert all(checkpoint.cell.metric == "revenue" for checkpoint in actual.frozen)
    finally:
        actual_analysis.close()
        expected_analysis.close()
        previous_analysis.close()


def test_unknown_evidence_requests_refuse_with_stable_coded_context() -> None:
    import copy
    import pickle

    analysis = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = cast(Any, held_source(analysis))
    provider = evidence_provider(source)
    try:
        with pytest.raises(CapabilityError) as raised:
            request_evidence(provider, object())
        refusal = raised.value
        assert refusal.code == "evidence.request_unsupported"
        assert refusal.context["requested"] == "object"
        assert refusal.context["offered"]
        with pytest.raises(TypeError):
            cast(Any, refusal.context)["requested"] = "changed"
        for restored in (pickle.loads(pickle.dumps(refusal)), copy.deepcopy(refusal)):
            assert type(restored) is type(refusal)
            assert restored.code == refusal.code
            assert restored.context == refusal.context
    finally:
        analysis.close()


def test_undeclared_unit_rows_refuse_before_custom_provider_reads() -> None:

    analysis = Analysis.from_unit_summary(
        unit_summary_table(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = cast(Any, held_source(analysis))
    metric = source.context.metrics[0]

    class Provider:
        def __init__(self) -> None:
            self.evidence: EvidenceCapabilities = replace(
                source.evidence,
                unit_rows=False,
                count_grains=frozenset({"assignment"}),
                populations=frozenset({"assigned"}),
            )
            self.context = source.context
            self.read = False

        def moments(self, request):
            return tuple(source.moments(request.metric, grain=request.grain, by=request.by))

        def unit_rows(self, request):
            self.read = True
            return source.unit_frame(request.metric, covariates=request.covariates)

        def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer:
            return MethodFitAnswer("unavailable")

        def breakout_method_moments(self, request):
            return source.breakout_moments_for_method(
                request.metric, request.method, by=(request.breakout.property,)
            )

        def compliance(self, request: ComplianceRequest):
            return source.compliance_summary(request.design, as_of=request.as_of)

        def compliance_series(self, request: ComplianceSeriesRequest):
            return ()

        def counts(self, request: CountsRequest) -> CountEvidence:
            self.read = True
            if request.kind == "unit":
                return CountEvidence("assigned", "unit", source.unit_counts())
            return CountEvidence("assigned", request.kind, {})

        def capture_sequential(self, request: SequentialCaptureRequest):
            return source.capture_sequential(
                finalized=request.finalized, as_of=request.as_of, previous=request.previous
            )

        def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
            return EvidenceScopeResult(None, request)

    provider = Provider()
    with pytest.raises(CapabilityError) as exc_info:
        request_evidence(provider, UnitRowsRequest(metric))

    assert exc_info.value.code == "evidence.unit_rows_unavailable"
    assert provider.read is False
    with pytest.raises(CapabilityError) as counts_refused:
        request_evidence(provider, CountsRequest("assignment", population="triggered"))
    assert counts_refused.value.code == "evidence.operation_unavailable"
    assert provider.read is False
    analysis.close()


def test_temporal_compliance_requests_refuse_total_only_provider_before_reads() -> None:
    from contextlib import nullcontext
    from datetime import date

    from tests.source_conformance import encouragement_panel_source

    source = encouragement_panel_source()

    class Provider:
        context = source.context
        evidence = replace(source.evidence, compliance_axis="total")

        def __init__(self) -> None:
            self.read = False

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused primitive")

        def compliance(self, request: ComplianceRequest):
            self.read = True
            return source.compliance_summary(request.design, as_of=request.as_of)

        def compliance_series(self, request: ComplianceSeriesRequest):
            self.read = True
            return source.compliance_series(request.design)

        def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
            return EvidenceScopeResult(nullcontext(self), request)

    provider = Provider()
    requests = (
        ComplianceRequest(source.context.design, as_of=date(2026, 1, 2)),
        ComplianceSeriesRequest(source.context.design),
    )
    for request in requests:
        with pytest.raises(CapabilityError) as raised:
            request_evidence(cast(Any, provider), request)
        assert raised.value.code == "evidence.operation_unavailable"
        assert provider.read is False


@pytest.mark.integration
def test_cluster_count_cube_does_not_require_unit_or_assignment_counts(tmp_path) -> None:
    import pyarrow.parquet as pq

    from increment.frame import MetricSpec
    from tests.test_compliance_summary import compliance_missingness_source, design

    source = compliance_missingness_source(missing_treatment=False, clustered=True)
    path = tmp_path / "cluster-counts.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row.pop("assignment_counts", None)

    analysis = Analysis.from_moments(
        rows,
        metrics=[MetricSpec(name="full", type="mean")],
        design=design(),
    )
    cube = cast(Any, held_source(analysis))
    try:
        provider = evidence_provider(cube)
        actual = provider.counts(CountsRequest("cluster"))

        assert actual == CountEvidence("assigned", "cluster", source.cluster_counts())
        assignment_request = CountsRequest("assignment")
        assert provider.counts(assignment_request) == CountEvidence(
            "assigned", "assignment", source.cluster_counts()
        )
        assert request_evidence(provider, assignment_request) == provider.counts(assignment_request)

        with pytest.raises(CodedError) as missing_units:
            cube.unit_counts()
        assert missing_units.value.code == "source.moments.assignment_counts"
    finally:
        analysis.close()


@pytest.mark.integration
def test_cluster_count_cube_returns_exact_cluster_grain_counts(tmp_path) -> None:
    import pyarrow.parquet as pq

    from increment.frame import MetricSpec
    from tests.test_compliance_summary import compliance_missingness_source, design

    source = compliance_missingness_source(missing_treatment=False, clustered=True)
    path = tmp_path / "cluster-counts-with-assignment.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    analysis = Analysis.from_moments(
        rows,
        metrics=[MetricSpec(name="full", type="mean")],
        design=design(),
    )
    cube = cast(Any, held_source(analysis))
    try:
        expected = cube.cluster_counts()
        actual = evidence_provider(cube).counts(CountsRequest("cluster"))
        assert actual == CountEvidence("assigned", "cluster", expected, cube.unit_counts())
    finally:
        analysis.close()


@pytest.mark.integration
def test_unclustered_source_refuses_cluster_counts_before_reading() -> None:
    source = frame_summary_source()

    class CountingSource(EvidenceProvider):
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = source.evidence
            self.cluster_count_reads = 0

        def cluster_counts(self):
            self.cluster_count_reads += 1
            return source.cluster_counts()

        def counts(self, request):
            require_evidence(self, request=request)
            return source.cluster_counts()

    counting_source = CountingSource()
    with pytest.raises(CapabilityError) as raised:
        evidence_provider(counting_source).counts(CountsRequest("cluster"))
    assert raised.value.code == "evidence.cluster_unavailable"
    assert counting_source.cluster_count_reads == 0


@pytest.mark.integration
def test_triggered_artifact_refuses_unavailable_cluster_counts_before_selecting_view(
    tmp_path, monkeypatch
) -> None:
    from increment.query.artifact_contract import ArtifactContractError
    from tests.analysis_factory import held_source
    from tests.source_conformance import unit_day_artifact_analysis

    analysis = unit_day_artifact_analysis(tmp_path)
    source = cast(Any, held_source(analysis))

    def invalid_trigger_view():
        raise ArtifactContractError("artifact.extension.invalid", "invalid trigger membership")

    monkeypatch.setattr(source, "triggered_source", invalid_trigger_view)
    try:
        with pytest.raises(CapabilityError) as refused:
            evidence_provider(source).counts(CountsRequest("cluster", population="triggered"))
        assert refused.value.code == "evidence.cluster_unavailable"
    finally:
        analysis.close()


@pytest.mark.integration
def test_unadjusted_breakout_moments_use_the_declared_method_family() -> None:
    import pyarrow as pa

    from increment.frame import MetricSpec, from_unit_summary
    from increment.semantics.models import CupedCovariate

    declared_method = Method.cuped_multicovariate(
        covariates=(CupedCovariate(name="pre", kind="continuous"),)
    )
    source = from_unit_summary(
        pa.table(
            {
                "unit": ["c-a", "t-a", "c-b", "t-b"],
                "arm": ["control", "treatment", "control", "treatment"],
                "segment": ["a", "a", "b", "b"],
                "region": ["x", "x", "y", "y"],
                "pre": [0.1, 0.2, 0.3, 0.4],
                "revenue": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", decision_method=declared_method)],
        breakouts=("segment", "region"),
    )
    metric = source.context.metrics[0]
    method = Method.unadjusted()
    request = MomentsRequest(metric, method=method, by=("segment",))
    with pytest.raises(CapabilityError) as unsupported:
        evidence_provider(source).moments(request)
    assert unsupported.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as unsupported:
        evidence_provider(source).moments(
            MomentsRequest(metric, method=method, by=("segment", "region"))
        )
    assert unsupported.value.code == "evidence.operation_unavailable"


@pytest.mark.integration
def test_assignment_counts_use_randomization_cluster_counts() -> None:
    import pyarrow as pa

    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "c3", "t1", "t2", "t3", "t4"],
                "arm": [
                    "control",
                    "control",
                    "control",
                    "treatment",
                    "treatment",
                    "treatment",
                    "treatment",
                ],
                "cluster": ["c1", "c1", "c2", "t1", "t1", "t2", "t2"],
                "revenue": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        ),
        unit="unit",
        group="arm",
        cluster="cluster",
        control="control",
        metrics={"revenue": "mean"},
    )
    source = cast(Any, held_source(analysis))
    try:
        cluster_counts = evidence_provider(source).counts(CountsRequest("cluster"))
        assert cluster_counts == CountEvidence(
            "assigned", "cluster", source.cluster_counts(), analysis.unit_counts()
        )
        assignment_counts = evidence_provider(source).counts(CountsRequest("assignment"))
        assert assignment_counts == CountEvidence("assigned", "assignment", source.cluster_counts())
    finally:
        analysis.close()


@pytest.mark.integration
def test_unit_rows_supports_custom_source_without_outcome_stage_parameter() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]
    expected = source.unit_frame(metric)

    class CustomSource(EvidenceProvider):
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = source.evidence

        def unit_rows(self, request):
            return source.unit_frame(request.metric, covariates=request.covariates)

    actual = evidence_provider(CustomSource()).unit_rows(UnitRowsRequest(metric))

    assert _frame_rows(actual) == _frame_rows(expected)


@pytest.mark.integration
def test_breakout_method_moments_preserve_rows_and_refused_cells() -> None:
    import pyarrow as pa

    from increment import BreakoutIdentity, BreakoutMethodMomentsRequest
    from increment.frame import from_unit_summary
    from increment.semantics.models import CupedCovariate

    source = from_unit_summary(
        pa.table(
            {
                "unit": ["c-a", "t-a", "c-b", "t-b"],
                "arm": ["control", "treatment", "control", "treatment"],
                "segment": ["a", "a", "b", "b"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
                "pre": [1.0, None, 2.0, 3.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=("segment",),
    )
    metric = source.context.metrics[0]
    method = Method.cuped(
        covariates=CupedCovariate(name="pre", kind="continuous", missing="refuse")
    )
    request = BreakoutMethodMomentsRequest(
        metric=metric,
        method=method,
        breakout=BreakoutIdentity("segment", None),
        population="assigned",
    )

    expected = source.breakout_moments_for_method(metric, method, by=("segment",))
    actual = evidence_provider(source).breakout_method_moments(request)

    assert actual == expected
    assert [(row["group_id"], row["ref_y"], row["segment"]) for row in actual.rows] == [
        ("control", 3.0, "b"),
        ("treatment", 4.0, "b"),
    ]
    assert [(dimension, value, spec.code) for dimension, value, spec, _ in actual.refusals] == [
        ("segment", "a", "frame.validation.metric_covariate_column")
    ]


@pytest.mark.integration
def test_breakout_method_moments_refuse_another_source_before_reading(tmp_path) -> None:
    from increment import BreakoutIdentity, BreakoutMethodMomentsRequest
    from tests.analysis_factory import held_source
    from tests.source_conformance import definitions_source

    analysis, declared_breakout = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    reads = 0

    class CountingSource:
        def __init__(self, inner):
            self.inner = inner
            self.context = inner.context
            self.evidence = inner.evidence

        def moments(self, request):
            return evidence_provider(self.inner).moments(request)

        def unit_rows(self, request):
            return evidence_provider(self.inner).unit_rows(request)

        def method_fit(self, request):
            return evidence_provider(self.inner).method_fit(request)

        def compliance(self, request):
            return evidence_provider(self.inner).compliance(request)

        def compliance_series(self, request):
            return evidence_provider(self.inner).compliance_series(request)

        def counts(self, request):
            return evidence_provider(self.inner).counts(request)

        def capture_sequential(self, request):
            return evidence_provider(self.inner).capture_sequential(request)

        def scope(self, request):
            return evidence_provider(self.inner).scope(request)

        def breakout_method_moments(self, request):
            nonlocal reads
            answer = evidence_provider(self.inner).breakout_method_moments(request)
            reads += 1
            return answer

    try:
        metric = source.context.metrics[0]
        provider = evidence_provider(CountingSource(source))
        method = Method.unadjusted()
        wrong_source = "src_b" if declared_breakout.source != "src_b" else "src_c"
        request = BreakoutMethodMomentsRequest(
            metric=metric,
            method=method,
            breakout=BreakoutIdentity("country", wrong_source),
        )

        with pytest.raises(CapabilityError) as unsupported:
            provider.breakout_method_moments(request)

        assert unsupported.value.code == "evidence.breakout_unavailable"
        assert reads == 0

        matching = provider.breakout_method_moments(
            BreakoutMethodMomentsRequest(
                metric=metric,
                method=method,
                breakout=BreakoutIdentity("country", declared_breakout.source),
            )
        )
        assert reads == 1
        assert {(row["group_id"], row["country"]) for row in matching.rows} == {
            ("control", "CA"),
            ("control", "US"),
            ("treatment", "CA"),
            ("treatment", "US"),
        }
    finally:
        analysis.close()


@pytest.mark.integration
def test_scoped_breakout_moments_reject_another_source_before_reading(
    tmp_path, monkeypatch
) -> None:
    from increment import BreakoutIdentity, BreakoutMethodMomentsRequest
    from tests.analysis_factory import held_source
    from tests.source_conformance import definitions_source

    analysis, declared_breakout = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    reads = 0

    try:
        metric = source.context.metrics[0]
        scoped = source.breakout_source(declared_breakout, metrics=(metric,))
        original = scoped.breakout_moments_for_method

        def count_reads(*args, **kwargs):
            nonlocal reads
            reads += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(scoped, "breakout_moments_for_method", count_reads)
        provider = evidence_provider(scoped)
        wrong_source = "src_b" if declared_breakout.source != "src_b" else "src_c"
        request = BreakoutMethodMomentsRequest(
            metric=metric,
            method=Method.unadjusted(),
            breakout=BreakoutIdentity("country", wrong_source),
        )

        with pytest.raises(CapabilityError) as unsupported:
            provider.breakout_method_moments(request)

        assert unsupported.value.code == "evidence.breakout_unavailable"
        assert reads == 0
    finally:
        analysis.close()


@pytest.mark.integration
def test_scoped_breakout_unit_rows_reject_another_source(tmp_path) -> None:
    from increment import BreakoutIdentity, BreakoutUnitRowsRequest
    from tests.analysis_factory import held_source
    from tests.source_conformance import definitions_source

    analysis, declared_breakout = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    try:
        metric = source.context.metrics[0]
        scoped = source.breakout_source(declared_breakout, metrics=(metric,))
        wrong_source = "src_b" if declared_breakout.source != "src_b" else "src_c"
        request = BreakoutUnitRowsRequest(
            metric=metric,
            breakout=BreakoutIdentity("country", wrong_source),
        )

        with pytest.raises(CapabilityError) as refused:
            request_evidence(evidence_provider(scoped), request)
        assert refused.value.code == "evidence.breakout_unavailable"
    finally:
        analysis.close()


def test_legacy_source_without_evidence_refuses_before_reading() -> None:
    source = frame_summary_source()

    class LegacySource:
        def __init__(self, inner):
            self._inner = inner
            self.reads = 0

        def __getattr__(self, name):
            if name == "evidence":
                raise AttributeError(name)
            if name == "moments":
                self.reads += 1
            return getattr(self._inner, name)

    legacy = LegacySource(source)
    with pytest.raises(CapabilityError) as unavailable:
        evidence_provider(legacy)
    assert unavailable.value.code == "evidence.provider_unavailable"
    assert legacy.reads == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("dimensioned_method", "evidence.operation_unavailable"),
        ("cluster", "evidence.cluster_unavailable"),
        ("unit_breakout", "evidence.breakout_unavailable"),
        ("breakout_slices", "evidence.breakout_unavailable"),
        ("breakout_method", "evidence.breakout_unavailable"),
    ],
)
def test_refusals_precede_population_or_source_reads(
    tmp_path, kind, expected_code, monkeypatch
) -> None:
    from increment import (
        BreakoutIdentity,
        BreakoutMethodMomentsRequest,
        BreakoutSlicesRequest,
        BreakoutUnitRowsRequest,
    )

    analysis, declared_breakout = definitions_source(
        tmp_path, trigger=True, partial_trigger=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    metric = source.context.metrics[0]
    wrong_source = "src_b" if declared_breakout.source != "src_b" else "src_c"
    identity = BreakoutIdentity(declared_breakout.property, wrong_source)
    reads = 0

    def fail_before_refusal(*args, **kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("source read before request refusal")

    for name in (
        "breakout_source",
        "breakout_sources",
        "moments",
        "moments_for_method",
        "triggered_source",
        "unit_frame",
    ):
        monkeypatch.setattr(source, name, fail_before_refusal)
    if kind == "dimensioned_method":
        request = MomentsRequest(
            metric,
            method=Method.unadjusted(),
            by=(declared_breakout.property,),
            population="triggered",
        )
    elif kind == "cluster":
        request = CountsRequest("cluster", population="triggered")
    elif kind == "unit_breakout":
        request = BreakoutUnitRowsRequest(metric, identity)
    elif kind == "breakout_slices":
        request = BreakoutSlicesRequest(
            breakouts=(identity,),
            metrics=(metric,),
            population="triggered",
        )
    else:
        request = BreakoutMethodMomentsRequest(
            metric=metric,
            method=Method.unadjusted(),
            breakout=identity,
            population="triggered",
        )

    try:
        with pytest.raises(CapabilityError) as refused:
            request_evidence(evidence_provider(source), request)
        assert refused.value.code == expected_code
        assert reads == 0
    finally:
        analysis.close()


def test_custom_compound_requests_use_primitive_reads_inside_provider_scope() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]
    identity = BreakoutIdentity("country", None)
    events: list[object] = []
    unit_frame = object()
    scope_configs: list[object] = []

    class Provider:
        evidence = replace(
            source.evidence,
            grains=frozenset({"total", "daily"}),
            breakouts=("country",),
            unit_rows=True,
            populations=frozenset({"assigned", "triggered"}),
        )
        context = source.context

        def moments(self, request: MomentsRequest):
            events.append(("moments", request.population))
            return (
                {
                    "ds": "d1",
                    "metric": metric.name,
                    "group_id": "control",
                    "n": 1,
                    "ref_y": 1.0,
                    "cy1": 0.0,
                    "cy2": 0.0,
                },
            )

        def unit_rows(self, request: UnitRowsRequest):
            events.append(("unit_rows", request.covariates))
            return unit_frame

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused primitive")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused primitive")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused primitive")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused primitive")

        def counts(self, request: CountsRequest):
            raise AssertionError("unused primitive")

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused primitive")

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                scope_configs.append(request.configs)
                events.append(("enter", request.population, request.breakouts))
                try:
                    yield self
                finally:
                    events.append(("exit", request.population, request.breakouts))

            return EvidenceScopeResult(scoped(), request)

    provider = Provider()
    assert isinstance(provider, EvidenceProvider)

    day_axis = request_evidence(
        provider,
        DayAxisRequest(
            caller="run_daily",
            grain="daily",
            metrics=(metric,),
            dimension=None,
            population="triggered",
        ),
    )
    assert events == []
    tuple(cast(Any, day_axis).slices)
    assert events == [
        ("enter", "triggered", ()),
        ("moments", "triggered"),
        ("exit", "triggered", ()),
    ]

    events.clear()
    breakout_rows = request_evidence(
        provider, BreakoutUnitRowsRequest(metric=metric, breakout=identity)
    )
    assert cast(Any, breakout_rows).frame is unit_frame
    assert events == [
        ("enter", "assigned", (identity,)),
        ("unit_rows", ("country",)),
        ("exit", "assigned", (identity,)),
    ]

    events.clear()
    triggered_rows = request_evidence(
        provider,
        BreakoutUnitRowsRequest(
            metric=metric,
            breakout=identity,
            population="triggered",
        ),
    )
    assert cast(Any, triggered_rows).frame is unit_frame
    assert events == [
        ("enter", "triggered", (identity,)),
        ("unit_rows", ("country",)),
        ("exit", "triggered", (identity,)),
    ]

    events.clear()
    scope_configs.clear()
    first_identity = BreakoutIdentity("country", "source_a")
    second_identity = BreakoutIdentity("country", "source_b")
    config_override = object()
    breakout_slices = cast(
        Any,
        request_evidence(
            provider,
            BreakoutSlicesRequest(
                breakouts=(first_identity, second_identity),
                metrics=(metric,),
                configs=(config_override,),
            ),
        ),
    )
    assert len(breakout_slices.slices) == 2
    assert events == []
    assert scope_configs == []

    request = DayAxisRequest(
        caller="run_daily",
        grain="daily",
        metrics=(metric,),
        dimension="country",
    )
    first_axis = request_evidence(breakout_slices.slices[0].source, request)
    tuple(cast(Any, first_axis).slices)
    second_axis = request_evidence(breakout_slices.slices[1].source, request)
    tuple(cast(Any, second_axis).slices)
    assert events == [
        ("enter", "assigned", (first_identity,)),
        ("moments", "assigned"),
        ("exit", "assigned", (first_identity,)),
        ("enter", "assigned", (second_identity,)),
        ("moments", "assigned"),
        ("exit", "assigned", (second_identity,)),
    ]
    assert scope_configs == [(config_override,), (config_override,)]


def test_custom_readouts_scope_day_axis_and_each_breakout_source() -> None:
    from increment.readouts import daily

    source = frame_summary_source()
    metric = source.context.metrics[0]
    first_identity = BreakoutIdentity("country", "source_a")
    second_identity = BreakoutIdentity("country", "source_b")
    events: list[object] = []

    class Provider:
        evidence = replace(
            source.evidence, grains=frozenset({"total", "daily"}), breakouts=("country",)
        )
        context = source.context

        def moments(self, request: MomentsRequest):
            events.append(("moments", request.by))
            return ()

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused primitive")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused primitive")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused primitive")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused primitive")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused primitive")

        def counts(self, request: CountsRequest):
            raise AssertionError("unused primitive")

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused primitive")

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                events.append(("enter", request.breakouts))
                try:
                    yield self
                finally:
                    events.append(("exit", request.breakouts))

            return EvidenceScopeResult(scoped(), request)

    provider = Provider()
    daily(provider, by=("country",))
    assert events == [
        ("enter", (BreakoutIdentity("country", None),)),
        ("moments", ("country",)),
        ("exit", (BreakoutIdentity("country", None),)),
    ]
    events.clear()

    slices = cast(
        Any,
        request_evidence(
            provider,
            BreakoutSlicesRequest(
                (first_identity, second_identity), metrics=(metric,), configs=(object(),)
            ),
        ),
    )
    for item in slices.slices:
        tuple(
            cast(
                Any,
                request_evidence(
                    item.source,
                    DayAxisRequest("run_daily", "daily", (metric,), dimension="country"),
                ),
            ).slices
        )
    assert events == [
        ("enter", (first_identity,)),
        ("moments", ("country",)),
        ("exit", (first_identity,)),
        ("enter", (second_identity,)),
        ("moments", ("country",)),
        ("exit", (second_identity,)),
    ]


@pytest.mark.integration
def test_triggered_multicovariate_fallback_reads_triggered_unit_rows() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]
    populations: list[str] = []

    class Provider:
        evidence = replace(
            source.evidence,
            unit_rows=True,
            populations=frozenset({"assigned", "triggered"}),
        )
        context = source.context

        def moments(self, request):
            return tuple(
                source.moments(
                    request.metric,
                    grain=request.grain,
                    by=request.by,
                    include_covariate=request.include_covariate,
                    completed_windows_only=request.completed_windows_only,
                )
            )

        def unit_rows(self, request):
            populations.append(request.population)
            return source.unit_frame(request.metric, covariates=request.covariates)

        def method_fit(self, request):
            return MethodFitAnswer("unavailable")

        def breakout_method_moments(self, request):
            raise AssertionError("unused")

        def compliance(self, request):
            raise AssertionError("unused")

        def compliance_series(self, request):
            raise AssertionError("unused")

        def counts(self, request):
            raise AssertionError("unused")

        def capture_sequential(self, request):
            raise AssertionError("unused")

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    from increment import CupedCovariate, Method
    from increment.evidence import MetricEvidenceRequest

    method = Method.cuped_multicovariate(
        covariates=(CupedCovariate(name="segment", kind="categorical"),)
    )
    request_evidence(
        Provider(),
        MetricEvidenceRequest(
            metric=metric,
            control_group="control",
            methods=(method,),
            covariates=(CupedCovariate(name="segment", kind="categorical"),),
            population="triggered",
        ),
    )
    assert populations == ["triggered"]


@pytest.mark.integration
def test_builtin_breakout_unit_rows_respect_triggered_population(tmp_path) -> None:
    from increment import BreakoutIdentity, BreakoutUnitRowsRequest

    source, identity = definitions_source(tmp_path, trigger=True, partial_trigger=True)
    metric = source.context.metrics[0]
    request = BreakoutUnitRowsRequest(
        metric=metric,
        breakout=BreakoutIdentity(identity.property, identity.source),
        population="triggered",
    )
    expected = source.triggered_source().unit_frame(metric, covariates=(identity.property,))

    actual = request_evidence(evidence_provider(source), request)

    assert _frame_rows(cast(Any, actual).frame) == _frame_rows(expected)


def test_custom_unit_counts_are_not_cluster_integrity_counts() -> None:
    import pandas as pd

    from increment.evidence import evidence_provider
    from increment.semantics.design import Randomized
    from tests.analysis_factory import held_source

    frame = pd.DataFrame(
        [
            {
                "unit": f"{arm}-{cluster}-{member}",
                "cluster_id": f"{arm}-{cluster}",
                "arm": arm,
                "revenue": float(member + (arm == "treatment")),
            }
            for arm, members in (("control", (10,) * 22), ("treatment", (1,) * 22))
            for cluster, count in enumerate(members)
            for member in range(count)
        ]
    )
    original = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        cluster="cluster_id",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics={"revenue": "mean"},
    )
    source = cast(Any, held_source(original))

    class Provider:
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = replace(source.evidence, count_grains=frozenset({"unit"}))

        def moments(self, request):
            return evidence_provider(source).moments(request)

        def unit_rows(self, request):
            return evidence_provider(source).unit_rows(request)

        def method_fit(self, request):
            return evidence_provider(source).method_fit(request)

        def breakout_method_moments(self, request):
            return evidence_provider(source).breakout_method_moments(request)

        def compliance(self, request):
            return evidence_provider(source).compliance(request)

        def compliance_series(self, request):
            return evidence_provider(source).compliance_series(request)

        def counts(self, request):
            return CountEvidence("assigned", "unit", source.unit_counts())

        def capture_sequential(self, request):
            return evidence_provider(source).capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    analysis = Analysis._from_source(cast(Any, Provider()))
    try:
        results = analysis.run()
        assert results.metadata is not None
        (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
        assert integrity.status == "not_checked_missing_counts"
        assert integrity.observed is None
    finally:
        analysis.close()
        original.close()


def test_custom_triggered_counts_are_not_assigned_integrity_counts() -> None:
    import pandas as pd

    from increment.evidence import evidence_provider
    from increment.semantics.design import Randomized
    from tests.analysis_factory import held_source

    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(40)],
            "arm": ["control"] * 20 + ["treatment"] * 20,
            "revenue": [float(i % 7) for i in range(40)],
        }
    )
    original = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
        metrics={"revenue": "mean"},
    )
    source = cast(Any, held_source(original))
    provider = evidence_provider(source)

    class Provider:
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = replace(
                source.evidence,
                count_grains=frozenset({"triggered"}),
                populations=frozenset({"assigned", "triggered"}),
            )

        def moments(self, request):
            return provider.moments(request)

        def unit_rows(self, request):
            return provider.unit_rows(request)

        def method_fit(self, request):
            return provider.method_fit(request)

        def breakout_method_moments(self, request):
            return provider.breakout_method_moments(request)

        def compliance(self, request):
            return provider.compliance(request)

        def compliance_series(self, request):
            return provider.compliance_series(request)

        def counts(self, request):
            # Balanced triggered counts say nothing about the assigned allocation.
            return CountEvidence("triggered", "triggered", {"control": 5, "treatment": 5})

        def capture_sequential(self, request):
            return provider.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    analysis = Analysis._from_source(cast(Any, Provider()))
    try:
        results = analysis.run()
        assert results.metadata is not None
        (integrity,) = next(iter(results.metadata.scope.by_source.values())).integrity
        assert integrity.status == "not_checked_missing_counts"
        assert integrity.observed is None
    finally:
        analysis.close()
        original.close()


def test_custom_provider_covariates_are_checked_from_declaration_before_reads() -> None:
    source = frame_summary_source()

    class Provider:
        evidence = replace(
            source.evidence,
            available_covariates=frozenset({"segment"}),
        )

    with pytest.raises(CapabilityError) as exc_info:
        require_evidence(Provider(), covariates=("missing",))

    assert exc_info.value.code == "evidence.covariate_unavailable"


def test_scoped_breakout_refuses_undeclared_unit_rows_before_reading() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]
    identity = BreakoutIdentity("country", None)
    reads: list[object] = []

    class Scoped:
        evidence = replace(
            source.evidence,
            breakouts=("country",),
            unit_rows=False,
            available_covariates=frozenset({"country"}),
        )
        context = source.context

        def unit_rows(self, request):
            reads.append(request)
            return source.unit_frame(metric)

        def moments(self, request):
            raise AssertionError("unused")

        def method_fit(self, request):
            raise AssertionError("unused")

        def breakout_method_moments(self, request):
            raise AssertionError("unused")

        def compliance(self, request):
            raise AssertionError("unused")

        def compliance_series(self, request):
            raise AssertionError("unused")

        def counts(self, request):
            raise AssertionError("unused")

        def capture_sequential(self, request):
            raise AssertionError("unused")

        def scope(self, request):
            raise AssertionError("unused")

    scoped = Scoped()

    class Provider:
        evidence = replace(
            source.evidence,
            breakouts=("country",),
            unit_rows=True,
            available_covariates=frozenset({"country"}),
        )
        context = source.context

        def moments(self, request):
            raise AssertionError("unused")

        def unit_rows(self, request):
            raise AssertionError("must use entered scope")

        def method_fit(self, request):
            raise AssertionError("unused")

        def breakout_method_moments(self, request):
            raise AssertionError("unused")

        def compliance(self, request):
            raise AssertionError("unused")

        def compliance_series(self, request):
            raise AssertionError("unused")

        def counts(self, request):
            raise AssertionError("unused")

        def capture_sequential(self, request):
            raise AssertionError("unused")

        def scope(self, request):
            @contextmanager
            def entered():
                yield scoped

            return EvidenceScopeResult(entered(), request)

    with pytest.raises(CapabilityError) as caught:
        request_evidence(Provider(), BreakoutUnitRowsRequest(metric=metric, breakout=identity))

    assert caught.value.code == "evidence.unit_rows_unavailable"
    assert reads == []


def test_triggered_breakout_fallback_requests_triggered_moments() -> None:
    import pyarrow as pa

    from increment.frame import from_unit_summary
    from increment.readouts import breakout

    source = from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "t1", "c2", "t2"],
                "arm": ["control", "treatment", "control", "treatment"],
                "country": ["US", "US", "US", "US"],
                "revenue": [1.0, 2.0, 1.5, 3.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"revenue": "mean"},
        breakouts=("country",),
    )
    metric = source.context.metrics[0]
    seen: list[str] = []
    triggered_rows = tuple(source.moments(metric, grain="total", by=("country",)))

    class Provider:
        evidence = replace(
            source.evidence,
            grains=frozenset({"total", "daily"}),
            breakouts=("country",),
            populations=frozenset({"assigned", "triggered"}),
            scalar=None,
        )
        context = source.context

        def moments(self, request):
            seen.append(request.population)
            return triggered_rows

        def unit_rows(self, request):
            raise AssertionError("unused")

        def method_fit(self, request):
            raise AssertionError("unused")

        def breakout_method_moments(self, request):
            raise AssertionError("unused")

        def compliance(self, request):
            raise AssertionError("unused")

        def compliance_series(self, request):
            raise AssertionError("unused")

        def counts(self, request):
            raise AssertionError("unused")

        def capture_sequential(self, request):
            raise AssertionError("unused")

        def scope(self, request):
            @contextmanager
            def entered():
                yield self

            return EvidenceScopeResult(entered(), request)

    result = breakout(
        Provider(), "country", metrics=(metric.name,), analysis_population="triggered"
    )

    expected = breakout(source, "country", metrics=(metric.name,))
    assert seen == ["triggered"]
    assert [(row.dimension_value, row.require_lift().value) for row in result] == [
        (row.dimension_value, row.require_lift().value) for row in expected
    ]


@pytest.mark.integration
def test_warehouse_fit_breakout_uses_per_cell_fit_answers_without_unit_rows(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import AnalysisPlan
    from increment.frame import MetricSpec
    from increment.readouts import breakout
    from increment.semantics.models import CupedCovariate

    method = Method.cuped_multicovariate(
        name="categorical",
        covariates=(CupedCovariate(name="segment", kind="categorical"),),
    )
    spec = MetricSpec(name="revenue", type="mean", decision_method=method)
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(16)],
                "arm": ["control"] * 8 + ["treatment"] * 8,
                "revenue": [float(i % 4) for i in range(8)]
                + [float(i % 4) + 1.0 for i in range(8)],
                "segment": ["a", "a", "b", "b"] * 4,
                "country": ["US", "CA"] * 8,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[spec],
        plan=AnalysisPlan(primary="revenue"),
        breakouts=("country",),
    )
    cube_path = tmp_path / "categorical.parquet"
    analysis.export(cube_path)
    replay = Analysis.from_moments(
        pq.read_table(cube_path).to_pylist(),
        metrics=[spec],
        control="control",
    )
    source = cast(Any, held_source(replay))
    source_provider = evidence_provider(source)
    metric = source.context.metrics[0]
    fit_requests: list[MethodFitRequest] = []
    unit_row_requests: list[UnitRowsRequest] = []

    class Provider:
        context = source.context
        evidence = replace(
            source.evidence,
            method_fit="warehouse",
            unit_rows=True,
            breakouts=("country",),
        )

        def moments(self, request):
            return tuple(
                {
                    "metric": metric.name,
                    "group_id": arm,
                    "country": country,
                    "n": 4,
                    "ref_y": value,
                    "cy1": 0.0,
                    "cy2": 0.0,
                }
                for country in ("CA", "US")
                for arm, value in (("control", 1.0), ("treatment", 2.0))
            )

        def unit_rows(self, request):
            unit_row_requests.append(request)
            return source_provider.unit_rows(request)

        def method_fit(self, request):
            fit_requests.append(request)
            answer = source_provider.method_fit(replace(request, cell=None))
            return MethodFitAnswer(
                "retained_categorical",
                cells=answer.cells,
                evidence_sha256=answer.evidence_sha256,
            )

        def breakout_method_moments(self, request):
            return source_provider.breakout_method_moments(request)

        def compliance(self, request):
            return source_provider.compliance(request)

        def compliance_series(self, request):
            return source_provider.compliance_series(request)

        def counts(self, request):
            return source_provider.counts(request)

        def capture_sequential(self, request):
            return source_provider.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        breakout(Provider(), "country", metrics=(metric.name,))
    finally:
        analysis.close()
        replay.close()

    assert len(fit_requests) == 2
    assert all(request.cell is not None for request in fit_requests)
    assert {request.cell.value for request in fit_requests if request.cell is not None} == {
        "CA",
        "US",
    }
    assert unit_row_requests == []


@pytest.mark.integration
def test_breakout_retained_cell_digest_changes_identity_not_estimates(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment import AnalysisPlan
    from increment.frame import MetricSpec
    from increment.readouts import breakout
    from increment.semantics.models import CupedCovariate

    method = Method.cuped_multicovariate(
        name="categorical",
        covariates=(CupedCovariate(name="segment", kind="categorical"),),
    )
    spec = MetricSpec(name="revenue", type="mean", decision_method=method)
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(16)],
                "arm": ["control"] * 8 + ["treatment"] * 8,
                "revenue": [float(i % 4) for i in range(8)]
                + [float(i % 4) + 1.0 for i in range(8)],
                "segment": ["a", "a", "b", "b"] * 4,
                "country": ["US", "CA"] * 8,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[spec],
        plan=AnalysisPlan(primary="revenue"),
        breakouts=("country",),
    )
    cube_path = tmp_path / "categorical.parquet"
    analysis.export(cube_path)
    replay = Analysis.from_moments(
        pq.read_table(cube_path).to_pylist(),
        metrics=[spec],
        control="control",
    )
    source = cast(Any, held_source(replay))
    source_provider = evidence_provider(source)
    metric = source.context.metrics[0]

    class Provider:
        context = source.context

        def __init__(self, digest: str) -> None:
            self.evidence = replace(
                source.evidence,
                method_fit="retained",
                unit_rows=True,
                breakouts=("country",),
            )
            self.digest = digest

        def moments(self, request):
            return tuple(
                {
                    "metric": metric.name,
                    "group_id": arm,
                    "country": country,
                    "n": 4,
                    "ref_y": value,
                    "cy1": 0.0,
                    "cy2": 0.0,
                }
                for country in ("CA", "US")
                for arm, value in (("control", 1.0), ("treatment", 2.0))
            )

        def unit_rows(self, request):
            raise AssertionError("retained categorical evidence should avoid unit rows")

        def method_fit(self, request):
            answer = source_provider.method_fit(replace(request, cell=None))
            return MethodFitAnswer(
                "retained_categorical",
                cells=answer.cells,
                evidence_sha256=self.digest,
            )

        def breakout_method_moments(self, request):
            return source_provider.breakout_method_moments(request)

        def compliance(self, request):
            return source_provider.compliance(request)

        def compliance_series(self, request):
            return source_provider.compliance_series(request)

        def counts(self, request):
            return source_provider.counts(request)

        def capture_sequential(self, request):
            return source_provider.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        first = breakout(Provider("a" * 64), "country", metrics=(metric.name,))
        second = breakout(Provider("b" * 64), "country", metrics=(metric.name,))
    finally:
        analysis.close()
        replay.close()

    assert [row.lift.value if row.lift is not None else None for row in first] == [
        row.lift.value if row.lift is not None else None for row in second
    ]
    assert [row.source_snapshot_id for row in first] != [row.source_snapshot_id for row in second]


@pytest.mark.integration
def test_custom_provider_daily_uses_typed_triggered_moments(tmp_path) -> None:
    from increment import Analysis

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    source = cast(Any, held_source(analysis))

    class Provider:
        evidence = source.evidence
        context = source.context
        artifact_experiment = analysis.experiment

        def moments(self, request: MomentsRequest):
            day_source = source.day_source(metrics=(request.metric,), population=request.population)
            return evidence_provider(day_source).moments(request)

        def triggered_observation_edges(self, metric: Any):
            return source.triggered_observation_edges(metric)

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            return evidence_provider(source).counts(request)

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                class Scoped(Provider):
                    triggered_observation_edges = None

                yield Scoped()

            return EvidenceScopeResult(scoped(), request)

    provider_analysis = Analysis._from_source(cast(Any, Provider()))
    try:
        expected = analysis.run_daily_lift(population="triggered")
        actual = provider_analysis.run_daily_lift(population="triggered")
    finally:
        analysis.close()

    assert [row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in actual] == [
        row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in expected
    ]


@pytest.mark.integration
def test_breakout_cell_without_fit_evidence_is_fitted_locally_from_unit_rows() -> None:
    import pyarrow as pa

    from increment import AnalysisPlan
    from increment.frame import MetricSpec
    from increment.readouts import breakout
    from increment.semantics.models import CupedCovariate

    method = Method.cuped_multicovariate(
        name="categorical",
        covariates=(CupedCovariate(name="segment", kind="categorical"),),
    )
    spec = MetricSpec(name="revenue", type="mean", decision_method=method)
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(32)],
                "arm": ["control"] * 16 + ["treatment"] * 16,
                "revenue": [float(i % 5) for i in range(16)]
                + [float(i % 5) + 1.5 for i in range(16)],
                "segment": ["a", "b", "b", "a"] * 8,
                "country": ["US", "CA"] * 16,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[spec],
        plan=AnalysisPlan(primary="revenue"),
        breakouts=("country",),
    )
    source = cast(Any, held_source(analysis))
    local = evidence_provider(source)
    unit_row_requests: list[object] = []

    class Provider:
        context = source.context
        evidence = replace(source.evidence, method_fit="warehouse", unit_rows=True)

        def moments(self, request):
            return local.moments(request)

        def unit_rows(self, request):
            unit_row_requests.append(request)
            return local.unit_rows(request)

        def method_fit(self, request):
            # Fit evidence covers no breakout cell, so every cell falls back to a local fit.
            return MethodFitAnswer("unavailable")

        def breakout_method_moments(self, request):
            return local.breakout_method_moments(request)

        def compliance(self, request):
            return local.compliance(request)

        def compliance_series(self, request):
            return local.compliance_series(request)

        def counts(self, request):
            return local.counts(request)

        def capture_sequential(self, request):
            return local.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        actual = breakout(Provider(), "country", metrics=("revenue",))
        expected = breakout(source, "country", metrics=("revenue",))
    finally:
        analysis.close()

    assert unit_row_requests
    lifts = {(row.dimension_value, row.method): row.lift for row in actual}
    assert lifts.keys() == {(row.dimension_value, row.method) for row in expected}
    for row in expected:
        assert row.lift is not None
        got = lifts[row.dimension_value, row.method]
        assert got is not None
        assert got.value == pytest.approx(row.lift.value)
        assert got.lb == pytest.approx(row.lift.lb)


@pytest.mark.integration
def test_custom_scope_yielding_builtin_loader_keeps_its_triggered_day_axis(tmp_path) -> None:
    from increment import Analysis

    # One treatment unit never triggers, so triggered and assigned day-axis rows differ.
    analysis, _ = definitions_source(
        tmp_path, trigger=True, partial_trigger=True, return_analysis=True
    )
    source = cast(Any, held_source(analysis))
    builtin = evidence_provider(source)

    class Provider:
        evidence = source.evidence
        context = source.context
        artifact_experiment = analysis.experiment

        def moments(self, request: MomentsRequest):
            raise AssertionError("the scoped built-in loader answers day-axis reads")

        def unit_rows(self, request: UnitRowsRequest):
            return builtin.unit_rows(request)

        def method_fit(self, request: MethodFitRequest):
            return builtin.method_fit(request)

        def breakout_method_moments(self, request: Any):
            return builtin.breakout_method_moments(request)

        def compliance(self, request: ComplianceRequest):
            return builtin.compliance(request)

        def compliance_series(self, request: ComplianceSeriesRequest):
            return builtin.compliance_series(request)

        def counts(self, request: CountsRequest):
            return builtin.counts(request)

        def capture_sequential(self, request: SequentialCaptureRequest):
            return builtin.capture_sequential(request)

        def scope(self, request: EvidenceScope):
            return builtin.scope(request)

    provider_analysis = Analysis._from_source(cast(Any, Provider()))
    try:
        expected = analysis.run_daily_lift(population="triggered")
        actual = provider_analysis.run_daily_lift(population="triggered")
    finally:
        analysis.close()

    assert [row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in actual] == [
        row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in expected
    ]


@pytest.mark.integration
def test_scoped_builtin_metadata_keeps_day_axis_breakouts(tmp_path) -> None:
    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    builtin = evidence_provider(source)
    provider = _forwarding_provider(source)
    provider.scope = builtin.scope
    try:
        result = cast(
            Any,
            request_evidence(
                provider,
                DayAxisRequest(
                    "run_daily",
                    "daily",
                    source.context.metrics,
                    dimension="country",
                ),
            ),
        )
        assert tuple(result.slices)
    finally:
        analysis.close()


def _forwarding_provider(source: Any, **attributes: Any) -> Any:
    builtin = evidence_provider(source)

    class Provider:
        context = source.context
        evidence = source.evidence

        def moments(self, request):
            return builtin.moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    provider = Provider()
    for name, value in attributes.items():
        setattr(provider, name, value)
    return provider


def _categorical_cuped_analysis() -> Any:
    import pyarrow as pa

    from increment import AnalysisPlan
    from increment.frame import MetricSpec
    from increment.semantics.models import CupedCovariate

    method = Method.cuped_multicovariate(
        name="categorical",
        covariates=(CupedCovariate(name="segment", kind="categorical"),),
    )
    return Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [f"u{i}" for i in range(32)],
                "arm": ["control"] * 16 + ["treatment"] * 16,
                "revenue": [float(i % 5) for i in range(16)]
                + [float(i % 5) + 1.5 for i in range(16)],
                "segment": ["a", "b", "b", "a"] * 8,
                "country": ["US", "CA"] * 16,
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="revenue", type="mean", decision_method=method)],
        plan=AnalysisPlan(primary="revenue"),
        breakouts=("country",),
    )


@pytest.mark.integration
def test_provider_declaring_breakouts_runs_breakouts_through_analysis() -> None:
    analysis = _categorical_cuped_analysis()
    try:
        expected = analysis.run_breakout()
        provider_analysis = Analysis._from_source(_forwarding_provider(held_source(analysis)))
        actual = provider_analysis.run_breakout()
    finally:
        analysis.close()

    assert expected
    assert {(row.dimension_value, row.method): row.lift for row in actual}.keys() == {
        (row.dimension_value, row.method) for row in expected
    }
    for row in expected:
        match = next(
            other
            for other in actual
            if (other.dimension_value, other.method) == (row.dimension_value, row.method)
        )
        assert (match.lift is None) == (row.lift is None)
        if row.lift is not None and match.lift is not None:
            assert match.lift.value == pytest.approx(row.lift.value)


@pytest.mark.integration
def test_provider_fit_requests_are_checked_against_its_own_declaration() -> None:
    import pyarrow as pa

    from increment.readouts import run

    analysis = _categorical_cuped_analysis()
    try:
        expected = analysis.run()
        # A provider may keep its own data under any attribute name, including `_source`.
        provider = _forwarding_provider(held_source(analysis), _source=pa.table({"revenue": [1.0]}))
        actual = run(provider)
    finally:
        analysis.close()

    assert [(row.method, row.group_id) for row in actual] == [
        (row.method, row.group_id) for row in expected
    ]
    for got, want in zip(actual, expected, strict=True):
        assert got.lift is not None and want.lift is not None
        assert got.lift.value == pytest.approx(want.lift.value)


@pytest.mark.integration
def test_custom_provider_sequential_run_uses_its_current_checkpoint(con) -> None:
    from datetime import date

    from increment import Analysis
    from increment.sequential_source import adopt_source_snapshot
    from tests.analysis_factory import held_source
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    provider_analysis = None
    try:
        source = cast(Any, held_source(native))
        snapshot = native.capture_sequential(finalized=True, as_of=date(2025, 7, 1))
        adopt_source_snapshot(source, snapshot)
        expected = list(native.run())
        provider_analysis = Analysis._from_source(
            _forwarding_provider(
                source,
                sequential_snapshot=lambda request: snapshot,
                close=lambda: None,
            )
        )
        from increment.readouts import run_scoped

        result = run_scoped(provider_analysis, provider_analysis.run)
        actual = list(result)
        assert actual == expected
        metadata = result.metadata
        assert metadata is not None
        assert metadata.coherence is None
        assert set(metadata.effective_modes.values()) == {"provider"}
    finally:
        if provider_analysis is not None:
            provider_analysis.close()
        native.close()


@pytest.mark.integration
def test_custom_provider_triggered_analysis_uses_declared_population(tmp_path) -> None:
    from increment import Analysis
    from tests.analysis_factory import held_source

    native, _ = definitions_source(
        tmp_path, trigger=True, partial_trigger=False, return_analysis=True
    )
    populations: list[str] = []
    operations_seen: set[str] = set()
    filtered_analysis = None
    provider_analysis = None
    try:
        expected = list(native.run(population="triggered"))
        source = cast(Any, held_source(native))
        provider = _forwarding_provider(
            source,
            artifact_experiment=native.experiment,
            close=lambda: None,
        )
        provider.source = source
        for operation in ("scope", "moments", "unit_rows"):
            original = getattr(provider, operation)

            def record_population(request, *, original=original, operation=operation):
                populations.append(request.population)
                operations_seen.add(operation)
                return original(request)

            setattr(provider, operation, record_population)
        provider_analysis = Analysis._from_source(provider, experiment=native.experiment)
        result = provider_analysis.run(population="triggered")
        actual = list(result)
        assert result.metadata is not None
        assert set(result.metadata.effective_modes.values()) == {"provider"}
        assert [row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in actual] == [
            row.model_dump(exclude={"source_snapshot_id", "family_id"}) for row in expected
        ]
        filtered_provider = _forwarding_provider(
            source,
            artifact_experiment=native.experiment,
            close=lambda: None,
        )
        filtered_provider.source = source
        filtered_provider.moments = lambda request: ()
        filtered_analysis = Analysis._from_source(filtered_provider, experiment=native.experiment)
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as filtered:
            list(filtered_analysis.run(population="triggered"))
        assert filtered.value.code == "readout.arms.no_treatment"
    finally:
        if provider_analysis is not None:
            provider_analysis.close()
        if filtered_analysis is not None:
            filtered_analysis.close()
        native.close()
    assert populations
    assert set(populations) == {"triggered"}
    assert "moments" in operations_seen


def test_warehouse_fit_keeps_unit_rows_for_uncovered_cuped_method() -> None:
    from types import SimpleNamespace

    from increment import CupedCovariate, Method
    from increment.evidence import MethodFitAnswer, MetricEvidenceRequest, request_evidence

    analysis = _categorical_cuped_analysis()
    source = cast(Any, held_source(analysis))
    builtin = evidence_provider(source)
    covariate = CupedCovariate(name="segment", kind="categorical")
    fitted_method = Method.cuped_multicovariate(name="warehouse", covariates=(covariate,))
    uncovered_method = Method.cuped_multicovariate(name="local", covariates=(covariate,))
    unit_requests: list[object] = []

    class Provider:
        context = source.context
        evidence = replace(source.evidence, method_fit="warehouse", unit_rows=True)

        def moments(self, request):
            return builtin.moments(request)

        def unit_rows(self, request):
            unit_requests.append(request)
            return builtin.unit_rows(request)

        def method_fit(self, request):
            if request.method.name == "warehouse":
                return MethodFitAnswer(
                    "warehouse",
                    envelope=SimpleNamespace(
                        fit=None, arm_order=("control", "treatment"), withheld=()
                    ),
                    fit_identity=("warehouse",),
                )
            return MethodFitAnswer("unavailable")

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        evidence = cast(
            Any,
            request_evidence(
                Provider(),
                MetricEvidenceRequest(
                    metric=source.context.metrics[0],
                    control_group="control",
                    methods=(fitted_method, uncovered_method),
                    covariates=(covariate,),
                ),
            ),
        )
    finally:
        analysis.close()

    assert evidence.unit_frame is not None
    assert unit_requests


def test_warehouse_fit_input_digest_uses_declared_unit_rows():
    from types import SimpleNamespace

    from increment import CupedCovariate, Method
    from increment.evidence import MethodFitAnswer, MetricEvidenceRequest, request_evidence

    analysis = _categorical_cuped_analysis()
    source = cast(Any, held_source(analysis))
    builtin = evidence_provider(source)
    segment = CupedCovariate(name="segment", kind="categorical")
    country = CupedCovariate(name="country", kind="categorical")

    class Provider:
        context = source.context
        evidence = replace(source.evidence, method_fit="warehouse", unit_rows=True)

        def moments(self, request):
            return builtin.moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return MethodFitAnswer(
                "warehouse",
                envelope=SimpleNamespace(fit=None, arm_order=("control", "treatment"), withheld=()),
            )

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            from contextlib import contextmanager

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        methods = (
            Method.cuped_multicovariate(name="segment-fit", covariates=(segment,)),
            Method.cuped_multicovariate(name="country-fit", covariates=(country,)),
        )
        request = MetricEvidenceRequest(
            metric=source.context.metrics[0],
            control_group="control",
            methods=methods,
            covariates=(segment, country),
        )
        evidence = cast(Any, request_evidence(Provider(), request))
        digests = [answer.evidence_sha256 for _, answer in evidence.method_fit_answers]
        assert all(digest is not None for digest in digests)
        assert digests[0] != digests[1]

        class NoUnitRowsProvider(Provider):
            evidence = replace(source.evidence, method_fit="warehouse", unit_rows=False)

        unavailable = cast(Any, request_evidence(NoUnitRowsProvider(), request))
        from increment.readouts import run

        unavailable_readout = run(NoUnitRowsProvider())
    finally:
        analysis.close()
    assert all(answer.evidence_sha256 is None for _, answer in unavailable.method_fit_answers)
    assert unavailable_readout.metadata is not None
    assert unavailable_readout.metadata.source_content_digest is None
    assert (
        unavailable_readout.metadata.source_content_digest_unavailable_reason
        == "method_fit_input_unavailable"
    )


def test_local_cuped_fallback_preserves_retained_categorical_components() -> None:
    from types import SimpleNamespace

    from increment import CupedCovariate, Method
    from increment.evidence import MethodFitAnswer, MetricEvidenceRequest, request_evidence

    analysis = _categorical_cuped_analysis()
    source = cast(Any, held_source(analysis))
    builtin = evidence_provider(source)
    covariate = CupedCovariate(name="segment", kind="categorical")
    decision = Method.cuped_multicovariate(name="retained", covariates=(covariate,))
    fallback = Method.cuped_multicovariate(name="fallback", covariates=(covariate,))

    class Provider:
        context = source.context
        evidence = replace(source.evidence, method_fit="retained", unit_rows=True)

        def moments(self, request):
            return builtin.moments(request)

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            if request.method.name == "retained":
                return MethodFitAnswer(
                    "retained_categorical",
                    cells=SimpleNamespace(arms=("control", "treatment"), counts=((2, 2), (2, 2))),
                    evidence_sha256="retained-categorical-evidence",
                )
            return MethodFitAnswer("unavailable")

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        evidence = cast(
            Any,
            request_evidence(
                Provider(),
                MetricEvidenceRequest(
                    metric=source.context.metrics[0],
                    control_group="control",
                    methods=(decision, fallback),
                    covariates=(covariate,),
                ),
            ),
        )
    finally:
        analysis.close()

    component = ("categorical_cells:retained", "retained-categorical-evidence", 1)
    assert component in evidence.evidence_components
    assert component in evidence.decision_evidence_components


@pytest.mark.integration
def test_failed_retained_categorical_fit_emits_unavailable_row() -> None:
    analysis = _categorical_cuped_analysis()
    source = cast(Any, held_source(analysis))
    failure = CodedError(
        "retained categorical fit failed",
        code="estimation.cuped.covariate_zero_variance",
        context={"method": "categorical", "weighted_var_x": 0.0},
    )
    provider = _forwarding_provider(source)
    provider.evidence = replace(source.evidence, method_fit="retained", unit_rows=True)
    provider.method_fit = lambda request: MethodFitAnswer("retained_categorical", failure=failure)
    provider.close = lambda: None
    provider_analysis = None
    try:
        provider_analysis = Analysis._from_source(provider)
        rows = list(provider_analysis.run(sensitivity_methods=[Method(name="unadjusted")]))
    finally:
        if provider_analysis is not None:
            provider_analysis.close()
        analysis.close()

    unavailable = [row for row in cast(Any, rows) if row.method == "categorical"]
    assert unavailable
    assert all(row.failure_code == failure.code for row in unavailable)


@pytest.mark.integration
def test_provider_checkpoint_wins_when_its_scope_yields_a_builtin_loader(con) -> None:
    from datetime import date

    from increment import Analysis
    from increment.sequential_source import adopt_source_snapshot
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    provider_analysis = None
    try:
        source = cast(Any, held_source(native))
        builtin = evidence_provider(source)
        # The provider owns the June 1 checkpoint; the loader it scopes to has moved to June 3.
        snapshot = native.capture_sequential(finalized=True, as_of=date(2025, 6, 1))
        later = native.capture_sequential(finalized=True, as_of=date(2025, 6, 3))
        adopt_source_snapshot(source, later)
        checkpoint_requests: list[object] = []

        def provider_checkpoint(request):
            checkpoint_requests.append(request)
            return snapshot

        provider_analysis = Analysis._from_source(
            _forwarding_provider(
                source,
                scope=builtin.scope,
                sequential_snapshot=provider_checkpoint,
                close=lambda: None,
            )
        )
        actual = list(provider_analysis.run())
        adopt_source_snapshot(source, snapshot)
        expected = list(native.run())
        assert checkpoint_requests
        assert actual == expected
    finally:
        if provider_analysis is not None:
            provider_analysis.close()
        native.close()


@pytest.mark.integration
def test_builtin_adapter_keeps_the_loader_checkpoint_for_sequential_readouts(con) -> None:
    from datetime import date

    from increment.readouts import run
    from increment.sequential_source import adopt_source_snapshot
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    try:
        source = cast(Any, held_source(native))
        adopt_source_snapshot(
            source, native.capture_sequential(finalized=True, as_of=date(2025, 7, 1))
        )
        assert list(run(evidence_provider(source))) == list(native.run())
    finally:
        native.close()


@pytest.mark.integration
def test_request_evidence_answers_sequential_snapshot_requests(con) -> None:
    from datetime import date

    from increment.evidence import SequentialSnapshotRequest, request_evidence
    from increment.sequential_source import adopt_source_snapshot
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    try:
        source = cast(Any, held_source(native))
        checkpoint = native.capture_sequential(finalized=True, as_of=date(2025, 7, 1))
        adopt_source_snapshot(source, checkpoint)
        answer = request_evidence(evidence_provider(source), SequentialSnapshotRequest())
        assert answer == native.sequential_snapshot()
    finally:
        native.close()


@pytest.mark.integration
def test_triggered_daily_lift_without_trigger_edges_refuses_before_reading(tmp_path) -> None:
    from increment import Analysis

    analysis, _ = definitions_source(tmp_path, trigger=True, return_analysis=True)
    source = cast(Any, held_source(analysis))
    reads: list[object] = []

    class Provider:
        evidence = source.evidence
        context = source.context
        artifact_experiment = analysis.experiment

        def moments(self, request: MomentsRequest):
            reads.append(request)
            raise AssertionError("refused requests read nothing")

        def unit_rows(self, request: UnitRowsRequest):
            reads.append(request)
            raise AssertionError("refused requests read nothing")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            return evidence_provider(source).counts(request)

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    assert analysis.experiment.trigger is not None
    provider_analysis = Analysis._from_source(cast(Any, Provider()))
    try:
        with pytest.raises(CodedError) as refused:
            provider_analysis.run_daily_lift(population="triggered")
    finally:
        analysis.close()

    assert refused.value.code == "evidence.operation_unavailable"
    assert reads == []


@pytest.mark.integration
def test_custom_daily_breakout_scopes_each_declared_source_separately(tmp_path) -> None:
    from dataclasses import replace as replace_dataclass

    from increment import Analysis
    from increment.semantics.models import Breakout

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    experiment = analysis.experiment.model_copy(
        update={
            "breakouts": (
                Breakout(property="country", source="web"),
                Breakout(property="country", source="app"),
            )
        }
    )
    scoped_breakouts: list[tuple[Any, ...]] = []

    class Provider:
        evidence = replace_dataclass(source.evidence, breakouts=("country",))
        context = source.context
        artifact_experiment = experiment

        def moments(self, request: MomentsRequest):
            return ()

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            return evidence_provider(source).counts(request)

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            scoped_breakouts.append(request.breakouts)

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    provider = Provider()
    provider_analysis = Analysis._from_source(cast(Any, provider))
    try:
        provider_analysis.run_daily(dimension="country")
        assert sorted(
            tuple((item.property, item.source) for item in items) for items in scoped_breakouts
        ) == [(("country", "app"),), (("country", "web"),)]
        scoped_breakouts.clear()
        from increment.readouts import run_scoped

        run_scoped(
            provider_analysis,
            lambda: provider_analysis.run_daily(dimension="country"),
            population="assigned",
            readouts=("daily",),
            include_breakouts=True,
        )
        scoped_identities = {
            tuple((item.property, item.source) for item in items) for items in scoped_breakouts
        }
        assert {(("country", "app"),), (("country", "web"),)} <= scoped_identities
        scoped_breakouts.clear()
        context_breakouts: list[tuple[Any, ...]] = []
        run_scoped(
            provider_analysis,
            lambda: None,
            population="assigned",
            readouts=("run",),
            on_context=lambda context: context_breakouts.append(context.scope.breakouts),
        )
        assert scoped_breakouts == [()]
        assert context_breakouts == [()]
    finally:
        analysis.close()


@pytest.mark.integration
def test_custom_daily_joint_grouping_refuses_dimensions_split_across_sources(
    tmp_path,
) -> None:
    from dataclasses import replace as replace_dataclass

    from increment.readouts import daily
    from increment.semantics.models import Breakout

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    experiment = analysis.experiment.model_copy(
        update={
            "breakouts": (
                Breakout(property="country", source="web"),
                Breakout(property="device", source="app"),
            )
        }
    )
    events: list[str] = []

    class Provider:
        evidence = replace_dataclass(source.evidence, breakouts=("country", "device"))
        context = source.context
        artifact_experiment = experiment

        def moments(self, request: MomentsRequest):
            events.append("read")
            return ()

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            raise AssertionError("unused")

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            events.append("scope")

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        with pytest.raises(CapabilityError) as error:
            daily(Provider(), by=("country", "device"))
        assert error.value.code == "evidence.joint_grouping_unavailable"
    finally:
        analysis.close()

    assert events == []


@pytest.mark.integration
def test_custom_daily_joint_grouping_keeps_dimensions_declared_on_same_source(
    tmp_path,
) -> None:
    from dataclasses import replace as replace_dataclass

    from increment.readouts import daily
    from increment.semantics.models import Breakout

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    experiment = analysis.experiment.model_copy(
        update={
            "breakouts": (
                Breakout(property="country", source="web"),
                Breakout(property="device", source="web"),
            )
        }
    )
    events: list[object] = []

    requested_dimensions: list[tuple[str, ...]] = []

    class Provider:
        evidence = replace_dataclass(source.evidence, breakouts=("country", "device"))
        context = source.context
        artifact_experiment = experiment

        def moments(self, request: MomentsRequest):
            requested_dimensions.append(request.by)
            events.append("read")
            return ()

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            raise AssertionError("unused")

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            events.append(
                ("scope", tuple((item.property, item.source) for item in request.breakouts))
            )

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        daily(Provider(), by=("country", "device"))
    finally:
        analysis.close()

    assert events == [("scope", (("country", "web"), ("device", "web"))), "read"]
    assert requested_dimensions
    assert all(set(dimensions) == {"country", "device"} for dimensions in requested_dimensions)


def test_triggered_day_axis_without_experiment_metadata_refuses_before_reading() -> None:
    from increment import Analysis

    source = frame_panel_source()
    context = replace(source.context, trigger_name="trigger")
    evidence = replace(
        source.evidence,
        populations=frozenset({"assigned", "triggered"}),
    )
    reads: list[object] = []
    provider = _forwarding_provider(source, context=context, evidence=evidence)
    provider.moments = lambda request: reads.append(request)
    analysis = Analysis._from_source(provider)

    with pytest.raises(CodedError) as refused:
        analysis.run_daily(population="triggered")

    assert refused.value.code == "evidence.operation_unavailable"
    assert refused.value.context["requested"] == "artifact_experiment"
    assert reads == []


def test_analysis_unit_counts_uses_declared_typed_count_evidence() -> None:
    from dataclasses import replace

    from increment import Analysis

    source = frame_panel_source()
    evidence = replace(source.evidence, operations=source.evidence.operations - {"unit_counts"})
    provider = _forwarding_provider(source, evidence=evidence)
    analysis = Analysis._from_source(provider)

    assert analysis.unit_counts() == source.unit_counts()


def test_analysis_close_accepts_provider_without_close_method() -> None:
    from increment import Analysis

    provider = _forwarding_provider(frame_panel_source())
    analysis = Analysis._from_source(provider)

    analysis.close()


def _assert_bound_daily_slice_reads_only_its_source(
    forwarding: Any, source: Any, experiment: Any
) -> None:
    from increment.semantics.models import Breakout

    bound_identity = BreakoutIdentity("country", "events")
    bound_definition = Breakout(property="country", source="events")
    bound_slices = cast(
        Any,
        request_evidence(
            forwarding,
            BreakoutSlicesRequest(
                breakouts=(bound_identity,),
                breakout_definitions=(bound_definition,),
                metrics=source.context.metrics,
                configs=source.context.configs,
            ),
        ),
    )
    bound_axis = request_evidence(
        bound_slices.slices[0].source,
        DayAxisRequest(
            "run_daily",
            "daily",
            source.context.metrics,
            dimension="country",
        ),
    )
    built_in_slices = tuple(cast(Any, bound_axis).slices)
    assert built_in_slices
    assert {item.source_name for item in built_in_slices} == {"events"}

    # A source-less declaration claims no other source, so a bound slice keeps it.
    sourceless = experiment.model_copy(
        update={"breakouts": (Breakout(property="country", source=None),)}
    )
    sourceless_axis = request_evidence(
        bound_slices.slices[0].source,
        DayAxisRequest(
            "run_daily",
            "daily",
            source.context.metrics,
            dimension="country",
            experiment=sourceless,
        ),
    )
    sourceless_slices = tuple(cast(Any, sourceless_axis).slices)
    assert sourceless_slices
    assert {item.source_name for item in sourceless_slices} == {"events"}

    inferred = cast(
        Any,
        request_evidence(
            forwarding,
            BreakoutSlicesRequest(
                breakouts=(BreakoutIdentity("country", None),),
                breakout_definitions=(bound_definition,),
                metrics=source.context.metrics,
                configs=source.context.configs,
            ),
        ),
    )
    inferred_axis = request_evidence(
        inferred.slices[0].source,
        DayAxisRequest(
            "run_daily",
            "daily",
            source.context.metrics,
            dimension="country",
            experiment=experiment,
        ),
    )
    inferred_slices = tuple(cast(Any, inferred_axis).slices)
    assert sorted((item.source_name, item.dimension) for item in inferred_slices) == [
        ("app", "country"),
        ("events", "country"),
    ]


@pytest.mark.integration
def test_builtin_scoped_daily_breakout_returns_each_source_row_once(tmp_path) -> None:
    import datetime as dt

    import ibis
    import yaml

    from increment import Analysis

    rows = [
        {
            "user_id": "u1",
            "ts": dt.datetime(2026, 1, 1, 9),
            "event": "assignment",
            "group_id": "control",
            "country_code": "US",
            "device_type": "mobile",
            "revenue": None,
        },
        {
            "user_id": "u1",
            "ts": dt.datetime(2026, 1, 1, 12),
            "event": "purchase",
            "group_id": None,
            "country_code": "US",
            "device_type": "mobile",
            "revenue": 1.0,
        },
    ]

    def fact_source(name: str, table: str) -> dict[str, Any]:
        return {
            "name": name,
            "sql": f"SELECT * FROM {table}",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": (
                [
                    {"name": "app_assignment", "column": None},
                    {"name": "app_purchase", "column": "revenue"},
                ]
                if name == "app"
                else [
                    {"name": "assignment", "column": None},
                    {"name": "purchase", "column": "revenue"},
                ]
            ),
            "properties": [
                {
                    "name": "country",
                    "column": "country_code",
                    "dtype": "string",
                    "as_of": "static",
                },
                {
                    "name": "device",
                    "column": "device_type",
                    "dtype": "string",
                    "as_of": "static",
                },
            ],
        }

    definitions = {
        "dialect": "duckdb",
        "fact_sources": [
            fact_source("events", "raw_events"),
            fact_source("app", "raw_app"),
        ],
        "exposures": [{"name": "assignment", "fact": "assignment"}],
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
                "name": "conformance",
                "exposure": "assignment",
                "unit": "user_id",
                "start": "2026-01-01",
                "end": "2026-01-03",
                "control_group": "control",
                "plan": {"primary": "revenue"},
                "breakouts": [
                    {"property": "country", "source": "events"},
                    {"property": "device", "source": "events"},
                    {"property": "country", "source": "app"},
                    {"property": "device", "source": "app"},
                ],
            },
        ],
    }
    defs_path = tmp_path / "two_sources.yaml"
    defs_path.write_text(yaml.safe_dump(definitions, sort_keys=False))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", obj=rows)
    con.create_table("raw_app", obj=rows)
    analysis = Analysis.from_definitions("conformance", defs_path, con)
    source = cast(Any, held_source(analysis))
    experiment = analysis.experiment
    builtin = evidence_provider(source)

    class Provider:
        evidence = replace(source.evidence, breakouts=("country", "device"))
        context = source.context
        artifact_experiment = experiment

        def __init__(self):
            self.active_scope: EvidenceScope | None = None
            self.rows_by_source = {
                "events": (
                    {
                        "ds": dt.date(2026, 1, 1),
                        "country": "US",
                        "device": "mobile",
                        "group_id": "control",
                        "revenue": 11.0,
                        "source": "events",
                    },
                    {
                        "ds": dt.date(2026, 1, 2),
                        "country": "US",
                        "device": "mobile",
                        "group_id": "treatment",
                        "revenue": 12.0,
                        "source": "events",
                    },
                ),
                "app": (
                    {
                        "ds": dt.date(2026, 1, 1),
                        "country": "CA",
                        "device": "tablet",
                        "group_id": "control",
                        "revenue": 21.0,
                        "source": "app",
                    },
                    {
                        "ds": dt.date(2026, 1, 2),
                        "country": "CA",
                        "device": "tablet",
                        "group_id": "treatment",
                        "revenue": 22.0,
                        "source": "app",
                    },
                ),
            }

        def moments(self, request):
            assert self.active_scope is not None
            sources = {identity.source for identity in self.active_scope.breakouts}
            if len(sources) != 1:
                return ()
            source_name = next(iter(sources))
            if source_name is None:
                return ()
            return self.rows_by_source[source_name]

        def unit_rows(self, request):
            return builtin.unit_rows(request)

        def method_fit(self, request):
            return builtin.method_fit(request)

        def breakout_method_moments(self, request):
            return builtin.breakout_method_moments(request)

        def compliance(self, request):
            return builtin.compliance(request)

        def compliance_series(self, request):
            return builtin.compliance_series(request)

        def counts(self, request):
            return builtin.counts(request)

        def capture_sequential(self, request):
            return builtin.capture_sequential(request)

        def scope(self, request: EvidenceScope):
            from contextlib import contextmanager

            from increment.evidence import EvidenceScopeResult

            delegated = builtin.scope(request)

            @contextmanager
            def scoped():
                with delegated.source:
                    self.active_scope = request
                    try:
                        yield self
                    finally:
                        self.active_scope = None

            return EvidenceScopeResult(scoped(), request)

    provider = Provider()
    forwarding = _forwarding_provider(source, scope=builtin.scope, artifact_experiment=experiment)
    forwarding_analysis = Analysis._from_source(forwarding)
    actual = forwarding_analysis.run_daily(dimension="country")
    identities = [
        (row.source, row.ds, row.dimension, row.dimension_value, row.group_id) for row in actual
    ]
    assert actual
    assert {row.source for row in actual} == {"events", "app"}
    assert len(identities) == len(set(identities))

    scoped_forwarding = _forwarding_provider(source, scope=builtin.scope)
    _assert_bound_daily_slice_reads_only_its_source(scoped_forwarding, source, experiment)

    declared = tuple(experiment.breakouts)
    identities = tuple(BreakoutIdentity(item.property, item.source) for item in declared)
    breakout_request = BreakoutSlicesRequest(
        breakouts=identities,
        breakout_definitions=declared,
        metrics=source.context.metrics,
        configs=source.context.configs,
    )
    original_breakouts = cast(Any, request_evidence(builtin, breakout_request))
    for identity, definition in zip(identities, declared, strict=True):
        acquired = cast(
            Any,
            request_evidence(
                builtin,
                replace(
                    breakout_request,
                    breakouts=(identity,),
                    breakout_definitions=(definition,),
                ),
            ),
        )
        expected = tuple(item for item in original_breakouts.slices if item.identity == identity)
        assert tuple(item.identity for item in acquired.slices) == tuple(
            item.identity for item in expected
        )
        assert acquired.slices[0].source.moments(
            source.context.metrics[0], by=(identity.property,)
        ) == expected[0].source.moments(source.context.metrics[0], by=(identity.property,))

    metric = source.context.metrics[0]

    custom_request = DayAxisRequest(
        caller="run_daily",
        grain="daily",
        metrics=(metric,),
        dimension=None,
        by=("country", "device"),
        experiment=experiment,
    )
    custom_slices = tuple(cast(Any, request_evidence(provider, custom_request)).slices)
    custom_rows = tuple(row for item in custom_slices for row in item.rows)
    expected_custom_rows = provider.rows_by_source["events"] + provider.rows_by_source["app"]
    assert [item.source_name for item in custom_slices] == ["events", "app"]
    assert custom_rows == expected_custom_rows
    custom_cell_ids = tuple(
        (row["source"], row["ds"], row["country"], row["device"], row["group_id"])
        for row in custom_rows
    )
    assert len(custom_cell_ids) == len(set(custom_cell_ids))
    assert [
        tuple((item.property, item.source) for item in day_slice.experiment.breakouts)
        for day_slice in custom_slices
    ] == [
        (("country", "events"), ("device", "events")),
        (("country", "app"), ("device", "app")),
    ]

    for grain, caller in (("daily", "run_daily"), ("asof", "run_asof")):
        day_request = DayAxisRequest(
            caller=caller,
            grain=grain,
            metrics=(metric,),
            dimension=None,
            by=("country", "device"),
            experiment=experiment,
        )
        unnormalized = tuple(cast(Any, request_evidence(provider, day_request)).slices)
        source_names = tuple(
            dict.fromkeys(item.source for item in declared if item.property in day_request.by)
        )
        assert source_names == ("events", "app")
        assert [item.source_name for item in unnormalized] == list(source_names)
        for source_name in source_names:
            child_experiment = experiment.model_copy(
                update={"breakouts": tuple(item for item in declared if item.source == source_name)}
            )
            child_request = replace(day_request, experiment=child_experiment)
            acquired = tuple(cast(Any, request_evidence(provider, child_request)).slices)
            expected = tuple(item for item in unnormalized if item.source_name == source_name)
            assert len(acquired) == len(expected) == 1
            assert acquired[0].rows == expected[0].rows

    analysis.close()


@pytest.mark.integration
def test_daily_moments_for_a_method_refuse_on_builtin_loaders(tmp_path) -> None:
    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    try:
        metric = source.context.metrics[0]
        with pytest.raises(CodedError) as refused:
            evidence_provider(source).moments(
                MomentsRequest(metric=metric, grain="daily", method=Method(name="unadjusted"))
            )
    finally:
        analysis.close()

    assert refused.value.code == "evidence.grain_unavailable"


@pytest.mark.integration
def test_provider_forwarding_its_checkpoint_through_request_evidence_runs_sequentially(
    con,
) -> None:
    from datetime import date

    from increment.evidence import SequentialSnapshotRequest
    from increment.sequential_source import adopt_source_snapshot
    from tests.sequential_cases import registered_native
    from tests.test_analysis_asof import _analysis_with_asof_events

    native = registered_native(_analysis_with_asof_events(con, sequential=True))
    provider_analysis = None
    try:
        source = cast(Any, held_source(native))
        builtin = evidence_provider(source)
        adopt_source_snapshot(
            source, native.capture_sequential(finalized=True, as_of=date(2025, 7, 1))
        )
        expected = list(native.run())

        def forwarded_checkpoint(request: SequentialSnapshotRequest):
            return request_evidence(builtin, request)

        provider_analysis = Analysis._from_source(
            _forwarding_provider(
                source,
                scope=builtin.scope,
                sequential_snapshot=forwarded_checkpoint,
                close=lambda: None,
            )
        )
        assert list(provider_analysis.run()) == expected
    finally:
        if provider_analysis is not None:
            provider_analysis.close()
        native.close()


@pytest.mark.integration
def test_day_axis_scopes_keep_requested_dimensions_the_experiment_does_not_declare(
    tmp_path,
) -> None:
    from increment.semantics.models import Breakout

    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    scoped_breakouts: list[tuple[Any, ...]] = []
    experiment = analysis.experiment.model_copy(
        update={"breakouts": (Breakout(property="country", source="events"),)}
    )

    class Provider:
        evidence = replace(source.evidence, breakouts=("country", "device"))
        context = source.context

        def moments(self, request: MomentsRequest):
            return ()

        def unit_rows(self, request: UnitRowsRequest):
            raise AssertionError("unused")

        def method_fit(self, request: MethodFitRequest):
            raise AssertionError("unused")

        def breakout_method_moments(self, request: Any):
            raise AssertionError("unused")

        def compliance(self, request: ComplianceRequest):
            raise AssertionError("unused")

        def compliance_series(self, request: ComplianceSeriesRequest):
            raise AssertionError("unused")

        def counts(self, request: CountsRequest):
            raise AssertionError("unused")

        def capture_sequential(self, request: SequentialCaptureRequest):
            raise AssertionError("unused")

        def scope(self, request: EvidenceScope):
            scoped_breakouts.append(tuple(item.property for item in request.breakouts))

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

    try:
        rows = cast(
            Any,
            request_evidence(
                cast(Any, Provider()),
                DayAxisRequest(
                    caller="run_daily",
                    grain="daily",
                    metrics=source.context.metrics,
                    dimension=None,
                    by=("country", "device"),
                    experiment=experiment,
                ),
            ),
        )
        list(rows.slices)
    finally:
        analysis.close()

    assert scoped_breakouts
    assert all(set(breakouts) == {"country", "device"} for breakouts in scoped_breakouts)


@pytest.mark.integration
def test_scope_result_source_is_evidence_provider(tmp_path):
    analysis, _ = definitions_source(tmp_path, return_analysis=True)
    source = cast(Any, held_source(analysis))
    provider = evidence_provider(source)
    metric = source.context.metrics[0]
    request = EvidenceScope(metrics=(metric,), execution=True)
    result = provider.scope(request)
    try:
        with result.source as scoped:
            assert isinstance(scoped, EvidenceProvider)
            assert scoped.moments(MomentsRequest(metric)) == tuple(source.moments(metric))
    finally:
        analysis.close()
