from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pyarrow as pa
import pytest

from increment import Method, MetricSpec, Scope, from_frame
from increment._evidence_types import DayAxisRequest, DayAxisRows
from increment.errors import IncrementWarning
from increment.semantics.models import MeanMetric
from tests.evidence_fixtures import (
    CaptureKey,
    EvidenceRecord,
    RequestCoordinates,
    assert_same_evidence,
    capture_acquired_evidence,
    method_identity,
)
from tests.warning_codes import warning_codes


def _key(metric="revenue", arm="control", method=None):
    method = method or Method(name="unadjusted")
    return CaptureKey(
        "randomized.unadjusted" if method.variance_reduction == "none" else "randomized.cuped",
        method_identity(method),
        metric,
        Scope(view="run", grain="total").hash(),
        RequestCoordinates("assigned", "total"),
        arm,
    )


def _summary(methods=()):
    table = pa.Table.from_pylist(
        [
            {
                "unit": f"{arm}{i}",
                "arm": arm,
                "revenue": 3.0 + i % 5 + (arm == "treatment"),
                "a": float(i % 3),
                "b": float(i % 4),
            }
            for arm in ("control", "treatment")
            for i in range(24)
        ]
    )
    options = (
        {} if not methods else {"decision_method": methods[0], "sensitivity_methods": methods[1:]}
    )
    return from_frame(
        table,
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="revenue", **options)],
    )


def _windowed_panel():
    from increment.semantics.models import AnalysisPlan
    from tests.parity_harness import dataset

    con = dataset.duckdb_connection()
    try:
        panel = dataset.sequential_unit_panel_frame(con)
    finally:
        con.disconnect()
    panel["converted"] = (panel["revenue"] > 0).astype(int)
    return from_frame(
        panel,
        shape="panel",
        unit="user_id",
        group="variant",
        date="date",
        exposure_date="exposure_date",
        observation_end="2025-01-20",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue", missing="zero", window_days=1, preferred_direction="increase"
            ),
            MetricSpec(
                name="purchase_rate",
                type="conversion",
                value_column="converted",
                window_days=1,
                preferred_direction="increase",
            ),
            MetricSpec(
                name="rps",
                type="ratio",
                numerator="revenue",
                denominator="sessions",
                missing="zero",
                window_days=1,
                preferred_direction="increase",
            ),
        ],
        plan=AnalysisPlan(q=0.10, secondaries=("revenue", "purchase_rate", "rps")),
    )


@pytest.mark.integration
@pytest.mark.slow
def test_mean_and_conversion_evidence_matches_across_sources(monkeypatch):
    from tests.parity_harness import dataset
    from tests.parity_harness.cases import _close_parity_binding, _fixed_horizon_case

    rows = dataset.event_rows()
    # Exactly at the excluded right edge, with nonzero values: a one-day
    # window accidentally extended to two must change the acquired evidence.
    rows += [
        {
            **row,
            "event": event,
            "event_at": datetime(2025, 1, 11, 9),
            "revenue": 100.0 if event == "purchase" else None,
            "sess": 1 if event == "session_end" else None,
        }
        for row in rows
        if row["event"] == "exposure"
        for event in ("purchase", "session_end")
    ]
    monkeypatch.setattr(dataset, "event_rows", lambda: [dict(row) for row in rows])

    case = _fixed_horizon_case(id="evidence-mean-conversion", breakout=False, include_cuped=False)
    expected = frozenset(
        [
            _key(metric, arm)
            for metric in ("revenue", "purchase_rate", "rps")
            for arm in ("control", "treatment")
        ]
        + [
            CaptureKey(
                "counts.arm_roster",
                "",
                "",
                _key().scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for arm in ("control", "treatment")
        ]
    )
    records = []
    for name, build in case.build.items():
        experiment, resolver = (_windowed_panel if name == "from_unit_panel" else build)()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IncrementWarning)
                record = capture_acquired_evidence(experiment, resolver, expected=expected)
            assert set(warning_codes(caught)) <= {"analysis.cuped.covariate_unused"}
            assert len(record.entries) == 8
            records.append(record)
        finally:
            _close_parity_binding((experiment, resolver))
    assert len(records) == 5
    for record in records[1:]:
        assert_same_evidence(records[0], record)
    payload = records[0].to_payload()
    assert payload["fixture_version"] == 1
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    # These nulls are estimator inputs, not absent fields or numerical zeros.
    assert records[0].entries[_key()]["moments"]["ref_den"] is None
    assert records[0].entries[_key()]["moments"]["n"] == 60


def test_day_axis_capture_keeps_slice_source_identity():
    from tests.evidence_fixtures.capture import _Capture

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    scope = Scope(view="daily", grain="daily", values=True)
    request = DayAxisRequest(
        caller="run_daily", grain="daily", metrics=(metric,), dimension="country"
    )
    slices = tuple(
        SimpleNamespace(
            rows=[
                {
                    "metric": "revenue",
                    "ds": "2025-01-01",
                    "group_id": "control",
                    "country": "us",
                }
            ],
            metrics=(metric,),
            view=None,
            dimension="country",
            source_name=source_name,
            observation_edges=None,
        )
        for source_name in ("events", "profiles")
    )
    expected = frozenset(
        CaptureKey(
            "day_axis",
            "",
            "revenue",
            scope.hash(),
            RequestCoordinates("assigned", "daily", "2025-01-01", ((f"{source}:country", "us"),)),
            "control",
        )
        for source in ("events", "profiles")
    )
    capture = _Capture((scope,), expected)
    result = DayAxisRows(object(), "native", iter(slices))
    recorded = capture.record("load_day_axis_rows", result, {"request": request, "scope": scope})
    tuple(recorded.slices)
    assert frozenset(key for key, _ in capture.entries) == expected


def test_method_fit_comparison_ignores_source_provenance_only():
    from increment._evidence_types import MetricEvidence
    from increment.evidence import MethodFitAnswer, MethodFitRequest
    from tests.evidence_fixtures.capture import _Capture

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    method = Method.cuped_multicovariate(name="mixed", covariates="a")
    scope = Scope(view="run", grain="total")
    request = MethodFitRequest(
        metric=metric,
        method=method,
        control="control",
        arm_order=("control", "treatment"),
        population="assigned",
    )
    expected = frozenset(
        CaptureKey(
            "randomized.cuped_multicovariate",
            method_identity(method),
            "revenue",
            scope.hash(),
            RequestCoordinates("assigned", "total"),
            arm,
        )
        for arm in ("control", "treatment")
    )

    def record(answer):
        capture = _Capture((scope,), expected)
        result = MetricEvidence(
            rows=({"group_id": "control", "n": 20}, {"group_id": "treatment", "n": 20}),
            unit_frame=None,
            observed_arms=frozenset(("control", "treatment")),
            n_treatment_arms=1,
            evidence_kind="frame",
            evidence_sha256="unused",
            evidence_count=40,
            decision_evidence_components=(),
            cluster=None,
            control_group="control",
            method_rows=(
                ("mixed", ({"group_id": "control", "n": 20}, {"group_id": "treatment", "n": 20})),
            ),
            method_fit_answers=((request, answer),),
        )

        def key(estimator, selected_method, selected_metric, arm, _row):
            return CaptureKey(
                estimator,
                method_identity(selected_method),
                selected_metric,
                scope.hash(),
                RequestCoordinates("assigned", "total"),
                arm,
            )

        capture.metric_records("load_metric_evidence", result, metric, (method,), key)
        return EvidenceRecord(capture.entries)

    left = record(
        MethodFitAnswer(
            origin="warehouse",
            envelope={"coefficients": {"a": 0.25}, "residual_variance": 3.0},
            fit_identity=("source-left", "fit"),
            evidence_sha256="left-digest",
        )
    )
    right = record(
        MethodFitAnswer(
            origin="retained_fitted",
            envelope={"coefficients": {"a": 0.25}, "residual_variance": 3.0},
            fit_identity=("source-right", "fit"),
            evidence_sha256="right-digest",
        )
    )
    assert_same_evidence(left, right)
    changed = record(
        MethodFitAnswer(
            origin="retained_fitted",
            envelope={"coefficients": {"a": 0.5}, "residual_variance": 3.0},
            fit_identity=("source-right", "fit"),
            evidence_sha256="right-digest",
        )
    )
    with pytest.raises(AssertionError):
        assert_same_evidence(left, changed)


@pytest.mark.integration
def test_mixed_cuped_secondary_fcr_reestimation_has_one_capture_boundary():
    from increment.semantics.models import AnalysisPlan
    from tests.analysis_factory import view_readout

    records = [
        {
            "unit": f"{arm}{index}",
            "arm": arm,
            "x": float(index % 4),
            "zero": float(index % 3),
            "borderline": 10.0 + 0.1 * index + (0.42 if arm == "treatment" else 0.0),
            "strong": 10.0 + 0.1 * index + (2.0 if arm == "treatment" else 0.0),
        }
        for arm in ("control", "treatment")
        for index in range(48)
    ]
    mixed = Method.cuped_multicovariate(name="mixed", covariates="x")
    default = Method(name="unadjusted")
    experiment, resolver = from_frame(
        pa.Table.from_pylist(records),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name=name,
                type="mean",
                decision_method=mixed if name == "strong" else None,
            )
            for name in ("zero", "borderline", "strong")
        ],
        plan=AnalysisPlan(secondaries=("zero", "borderline", "strong"), q=0.05),
    )
    scope_hash = Scope(view="run", grain="total").hash()
    expected = (
        frozenset(
            CaptureKey(
                "randomized.unadjusted" if name != "strong" else "randomized.cuped_multicovariate",
                method_identity(default if name != "strong" else mixed),
                name,
                scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for name in ("zero", "borderline", "strong")
            for arm in ("control", "treatment")
        )
        | frozenset(
            CaptureKey(
                "counts.arm_roster",
                "",
                "",
                scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for arm in ("control", "treatment")
        )
        | frozenset(
            CaptureKey(
                "units.randomized.cuped_multicovariate",
                method_identity(mixed),
                "strong",
                scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for arm in ("control", "treatment")
        )
    )
    try:
        capture_acquired_evidence(experiment, resolver, expected=expected)
        rows = view_readout(experiment, resolver)
    finally:
        resolver.close()
    strong = next(row for row in rows if row.metric == "strong")
    assert strong.family_size == 3
    assert strong.family_threshold == pytest.approx(0.05 / 3)

    assert strong.discovery is True


def test_cuped_companions_with_different_covariates_capture_separately():
    methods = (
        Method.cuped(name="first", covariates="a"),
        Method.cuped(name="second", covariates="b"),
    )
    experiment, resolver = _summary(methods)
    expected = frozenset(
        [_key(arm=arm, method=method) for method in methods for arm in ("control", "treatment")]
        + [
            CaptureKey(
                "counts.arm_roster",
                "",
                "",
                _key().scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for arm in ("control", "treatment")
        ]
    )
    try:
        record = capture_acquired_evidence(experiment, resolver, expected=expected)
    finally:
        resolver.close()
    assert method_identity(methods[0]) != method_identity(methods[1])
    assert method_identity(methods[0]) != method_identity(
        Method.cuped(name="first", covariates="b")
    )
    assert record.entries[_key(method=methods[0])]["moments"]["ref_x"] == 1.0
    assert record.entries[_key(method=methods[1])]["moments"]["ref_x"] == 1.5
    for method in methods:
        key = _key(method=method)
        original = record.entries[key]
        changed_payload = deepcopy(original)
        changed_payload["moments"]["ref_x"] = -10.0
        changed = EvidenceRecord([(key, changed_payload)])
        with pytest.raises(AssertionError):
            assert_same_evidence(EvidenceRecord([(key, original)]), changed)


def test_duplicate_capture_key_raises():
    with pytest.raises(AssertionError):
        EvidenceRecord([(_key(), {"n": 1}), (_key(), {"n": 2})])


def test_capture_roster_mismatch_fails():
    experiment, resolver = _summary()
    actual = frozenset(
        [_key(arm=arm) for arm in ("control", "treatment")]
        + [
            CaptureKey(
                "counts.arm_roster",
                "",
                "",
                _key().scope_hash,
                RequestCoordinates("assigned", "total"),
                arm,
            )
            for arm in ("control", "treatment")
        ]
    )
    try:
        assert (
            capture_acquired_evidence(experiment, resolver, expected=actual).entries.keys()
            == actual
        )
        for expected in (actual - {_key()}, actual | {replace(_key(), arm="missing")}):
            with pytest.raises(AssertionError) as raised:
                capture_acquired_evidence(experiment, resolver, expected=expected)
            assert raised.value.args[0] == {
                "missing": expected - actual,
                "unexpected": actual - expected,
            }
        with pytest.raises(AssertionError):
            capture_acquired_evidence(experiment, resolver, expected=frozenset())
    finally:
        resolver.close()
    with pytest.raises(AssertionError):
        assert_same_evidence(EvidenceRecord([]), EvidenceRecord([]))


def test_transcript_codec_rejects_an_unregistered_version():
    from tests.evidence_fixtures.transcript import Transcript

    payload = {
        "schema_version": 2,
        "baseline_revision": "7c324b66",
        "scenario_id": "scalar-checkpoint",
        "ingress": "from_moments",
        "runtime": {"python": "3.13.5", "platform": "test"},
        "provenance": {"kind": "historical-input"},
        "declaration_sha256": "unused",
        "input_sha256": "unused",
        "records": [],
    }
    with pytest.raises(ValueError, match="schema_version"):
        Transcript.from_payload(payload, expected_requests=())


@pytest.mark.integration
@pytest.mark.slow
def test_frozen_scalar_evidence_runs_mean_conversion_ratio_consumers():
    from pathlib import Path

    from increment._evidence_types import (
        AssignmentRequest,
        CountsRequest,
        MetricEvidenceRequest,
        MomentsRequest,
    )
    from increment.estimation.engine import estimate_lift
    from increment.evidence import evidence_provider
    from tests.analysis_factory import bound_source
    from tests.evidence_fixtures.transcript import Transcript, acquire_transcript
    from tests.parity_harness.cases import _close_parity_binding, _fixed_horizon_case

    case = _fixed_horizon_case(id="mean_ratio_conversion_cuped_fixed_horizon", breakout=False)
    experiment, resolver = case.build["from_moments"]()
    try:
        source = bound_source(resolver)
        metrics = {
            metric.name: metric
            for metric in source.context.metrics
            if metric.name in {"revenue", "purchase_rate", "rps"}
        }
        assert set(metrics) == {"revenue", "purchase_rate", "rps"}
        metric_requests = {
            name: MetricEvidenceRequest(metric=metric, control_group="control")
            for name, metric in metrics.items()
        }
        moments_requests = {name: MomentsRequest(metric=metric) for name, metric in metrics.items()}
        requests = (
            *metric_requests.values(),
            *moments_requests.values(),
            CountsRequest(kind="assignment"),
            AssignmentRequest(experiment="moments"),
        )
        live = acquire_transcript(
            evidence_provider(source),
            scenario_id=case.id,
            ingress="from_moments",
            requests=requests,
            baseline_revision="7c324b66",
            provenance={
                "kind": "baseline-input-capture",
                "scenario": case.id,
                "input": "case-owned fixed-horizon input and baseline M export",
            },
        )
        fixture = (
            Path(__file__).parent / "fixtures" / "evidence_identity" / "scalar_checkpoint.json"
        )
        frozen = Transcript.from_payload(
            json.loads(fixture.read_text()), expected_requests=requests
        )

        def evidence_record(transcript):
            return EvidenceRecord(
                (
                    CaptureKey(
                        record["request_type"],
                        record["request_type"],
                        record["request"].get("metric", {}).get("name", ""),
                        transcript.scenario_id,
                        RequestCoordinates(
                            record["request"].get("population", "assigned"),
                            record["request"].get("grain", "total"),
                        ),
                        str(index),
                    ),
                    record["answer"],
                )
                for index, record in enumerate(transcript.records)
            )

        assert_same_evidence(evidence_record(frozen), evidence_record(live))

        for name, metric in metrics.items():
            evidence = frozen.answer_for(metric_requests[name])
            expected_rows = {
                row["group_id"]: row for row in frozen.answer_for(moments_requests[name])
            }
            (result,) = estimate_lift([metric], evidence["rows"], control_group="control").results
            if name == "purchase_rate":
                control = expected_rows["control"]["successes"] / expected_rows["control"]["n"]
                treatment = (
                    expected_rows["treatment"]["successes"] / expected_rows["treatment"]["n"]
                )
            elif name == "rps":
                control = expected_rows["control"]["ref_y"] / expected_rows["control"]["ref_den"]
                treatment = (
                    expected_rows["treatment"]["ref_y"] / expected_rows["treatment"]["ref_den"]
                )
            else:
                control = expected_rows["control"]["ref_y"]
                treatment = expected_rows["treatment"]["ref_y"]
            assert result.lift is not None
            assert result.lift.value == pytest.approx(treatment / control - 1)

        assert frozen.answer_for(CountsRequest(kind="assignment"))["counts"]
        assert frozen.answer_for(AssignmentRequest(experiment="moments"))["arm_counts"]
    finally:
        _close_parity_binding((experiment, resolver))


def test_transcript_decoder_rejects_roster_and_answer_schema_drift():
    from copy import deepcopy

    from increment._evidence_types import CountsRequest
    from tests.evidence_fixtures.transcript import Transcript

    request = CountsRequest(kind="unit")
    transcript = Transcript(
        baseline_revision="7c324b66",
        scenario_id="scalar-checkpoint",
        ingress="from_moments",
        runtime={"python": "3.13.5", "platform": "test"},
        provenance={"authority": "historical-input"},
        records=(
            {
                "request_type": "CountsRequest",
                "request": {"kind": "unit", "population": "assigned"},
                "answer_type": "CountEvidence",
                "answer": {
                    "population": "assigned",
                    "grain": "unit",
                    "counts": {"control": 60, "treatment": 60},
                    "unit_counts": {},
                },
            },
        ),
    )
    payload = transcript.to_payload()
    Transcript.from_payload(payload, expected_requests=(request,))

    missing = deepcopy(payload)
    missing["records"] = []
    with pytest.raises(ValueError, match="roster mismatch"):
        Transcript.from_payload(missing, expected_requests=(request,))

    duplicate = deepcopy(payload)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="duplicate transcript request"):
        Transcript.from_payload(duplicate, expected_requests=(request,))

    unexpected = deepcopy(payload)
    unexpected["records"][0]["request"]["kind"] = "cluster"
    with pytest.raises(ValueError, match="roster mismatch"):
        Transcript.from_payload(unexpected, expected_requests=(request,))

    unknown_field = deepcopy(payload)
    unknown_field["records"][0]["answer"]["discarded"] = True
    with pytest.raises(ValueError, match="answer fields"):
        Transcript.from_payload(unknown_field, expected_requests=(request,))
