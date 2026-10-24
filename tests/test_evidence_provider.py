from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

import pytest

from increment import Analysis, EvidenceCapabilities, Method
from increment.errors import CapabilityError, CodedError
from increment.evidence import (
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountEvidence,
    CountsRequest,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitAnswer,
    MethodFitRequest,
    MomentsRequest,
    SequentialCaptureRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
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
            assert scope.scope == scope_request
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

        def __getattr__(self, name):
            if name == "assignment_counts":
                raise AttributeError(name)
            return getattr(source, name)

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

    analysis = registered_native(_analysis_with_asof_events(con, sequential=True))
    try:
        as_of = date(2025, 7, 1)
        expected = analysis.capture_sequential(finalized=True, as_of=as_of)
        actual = evidence_provider(held_source(analysis)).capture_sequential(
            SequentialCaptureRequest(finalized=True, as_of=as_of)
        )
        assert actual == expected
        assert actual.states == expected.states
        assert actual.records == expected.records
        assert actual.frozen == expected.frozen
    finally:
        analysis.close()


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
            self.evidence: EvidenceCapabilities = replace(source.evidence, unit_rows=False)
            self.context = source.context
            self.read = False

        def breakout_evidence(self, request: Any) -> Any:
            raise NotImplementedError

        def day_axis(self, request: Any) -> Any:
            raise NotImplementedError

        def metric_evidence(self, request: Any) -> Any:
            raise NotImplementedError

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
    analysis.close()


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

    class CountingSource:
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = source.evidence
            self.cluster_count_reads = 0

        def cluster_counts(self):
            self.cluster_count_reads += 1
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

    class CustomSource:
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = source.evidence

        def __getattr__(self, name):
            return getattr(source, name)

        def unit_frame(self, metric, *, covariates=()):
            return source.unit_frame(metric, covariates=covariates)

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

    class CountingBreakoutSource:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def breakout_moments_for_method(self, *args, **kwargs):
            nonlocal reads
            reads += 1
            return self.inner.breakout_moments_for_method(*args, **kwargs)

    class CountingSource:
        def __init__(self, inner):
            self.inner = inner
            self.context = inner.context
            self.evidence = inner.evidence

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def breakout_source(self, *args, **kwargs):
            return CountingBreakoutSource(self.inner.breakout_source(*args, **kwargs))

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
            evidence_provider(scoped).breakout_evidence(request)
        assert refused.value.code == "evidence.breakout_unavailable"
    finally:
        analysis.close()


def test_legacy_source_without_evidence_uses_derived_declaration() -> None:
    source = frame_summary_source()
    metric = source.context.metrics[0]

    class LegacySource:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "evidence":
                raise AttributeError(name)
            return getattr(self._inner, name)

    legacy = LegacySource(source)
    provider = evidence_provider(legacy)

    assert provider.moments(MomentsRequest(metric)) == tuple(source.moments(metric))
    assert provider.evidence.kind == "moments"
    assert provider.evidence.grains == frozenset(source.capabilities)


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
