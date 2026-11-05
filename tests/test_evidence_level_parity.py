from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from functools import cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pyarrow as pa
import pytest

from increment import Method, MetricSpec, Scope, from_frame
from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment._evidence_types import DayAxisRequest, DayAxisRows
from increment.errors import CodedError, IncrementWarning
from increment.semantics.models import MeanMetric
from tests.evidence_fixtures import (
    CaptureKey,
    EvidenceRecord,
    RequestCoordinates,
    assert_same_evidence,
    capture_acquired_evidence,
    method_identity,
)
from tests.evidence_fixtures.scenarios import (
    acquire_scenario,
    observe_scenario_disposition,
    scenario_roster,
)
from tests.evidence_fixtures.transcript import (
    _REQUEST_CLASSES,
    Transcript,
    _digest,
    require_ancestor_revision,
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
    from increment._binding import binding_for
    from increment._evidence_types import MetricEvidenceRequest
    from increment.evidence import CountsRequest, evidence_provider, request_evidence
    from tests.evidence_fixtures.capture import _Capture
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
                source = binding_for(experiment, resolver).source
                provider = evidence_provider(source)
                captured = _Capture((Scope(view="run", grain="total"),), expected)
                for metric in source.context.metrics:
                    request = MetricEvidenceRequest(
                        metric=metric,
                        control_group="control",
                        methods=(Method(name="unadjusted"),),
                    )
                    captured.record(
                        "load_metric_evidence",
                        request_evidence(provider, request),
                        {"src": source, "request": request},
                    )
                request = CountsRequest(kind="unit")
                captured.record(
                    "load_counts",
                    request_evidence(provider, request),
                    {"src": source, "request": request},
                )
                record = EvidenceRecord(captured.entries)
                assert frozenset(record.entries) == expected
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


def _parity_revenue(arm: str, unit: int) -> float:
    """In-window revenue of unit *unit* in `dataset.event_rows`: a third buy nothing."""
    return 0.0 if unit % 3 == 0 else 5.0 + (arm == "treatment") + unit % 5


@pytest.mark.integration
def test_triggered_run_acquires_the_triggered_members_evidence():
    """A whole-window run of the triggered population requests triggered evidence and receives
    the triggered units' values: here the first half of each arm sees the surface at
    assignment, so its counts and moments differ from the assigned arm's."""
    import datetime as dt
    import statistics
    from dataclasses import dataclass, field

    from increment.artifacts import SourceSnapshotEvidence
    from increment.semantics.models import AnalysisPlan, Definitions
    from tests.analysis_factory import make_binding
    from tests.evidence_fixtures.capture import _Capture, _wrapped_producers
    from tests.parity_harness import dataset as ds
    from tests.parity_harness.cases import _close_parity_binding, _track_connection
    from tests.parity_harness.views import read

    @dataclass
    class _Acquisitions(_Capture):
        seen: list = field(default_factory=list)

        def record(self, name, result, arguments):
            self.seen.append((self.context(arguments).get("request"), result))
            return result

    per_arm, triggered = 24, range(12)
    rows = ds.event_rows(n_per_arm=per_arm)
    rows += [
        ds._row(f"{prefix}{unit}", ds._EXPOSURE_AT, "saw_surface")
        for prefix in ("c", "t")
        for unit in triggered
    ]
    payload = ds.definitions_dict(plan=AnalysisPlan(primary="revenue"))
    payload["fact_sources"][0]["facts"].append({"name": "saw_surface", "column": None})
    payload["exposures"].append({"name": "saw_surface", "fact": "saw_surface"})
    payload["experiments"][0]["trigger"] = "saw_surface"
    cutoff = dt.datetime(2025, 1, 19, 23, 59, tzinfo=dt.UTC)
    con = ds.duckdb_connection(rows)
    bound = _track_connection(
        make_binding(
            con,
            Definitions.model_validate(payload),
            experiment="exp",
            source_snapshot_evidence=SourceSnapshotEvidence(
                observation_cutoff_ts=cutoff, complete_through_by_feed={"events": cutoff}
            ),
        ),
        con,
    )
    acquisitions = _Acquisitions(
        (Scope(view="run", grain="total", population="triggered"),), frozenset()
    )
    try:
        with warnings.catch_warnings(), _wrapped_producers(acquisitions):
            warnings.simplefilter("ignore", IncrementWarning)
            readout = read(bound, "run", population="triggered", metrics=("revenue",))
    finally:
        _close_parity_binding(bound)

    expected = {
        arm: [_parity_revenue(arm, unit) for unit in triggered] for arm in ("control", "treatment")
    }
    assigned = {arm: [_parity_revenue(arm, unit) for unit in range(per_arm)] for arm in expected}
    assert all(
        statistics.fmean(expected[arm]) != statistics.fmean(assigned[arm]) for arm in expected
    )
    evidence = [
        (request, result)
        for request, result in acquisitions.seen
        if type(request).__name__ == "MetricEvidenceRequest"
    ]
    ((request, result),) = evidence
    moments = {row["group_id"]: row for row in result.rows}
    for arm, values in expected.items():
        mean = statistics.fmean(values)
        assert moments[arm]["n"] == len(values)
        assert moments[arm]["ref_y"] == pytest.approx(mean, rel=1e-9)
        assert moments[arm]["cy2"] == pytest.approx(
            sum((value - mean) ** 2 for value in values), rel=1e-9
        )
    counts = [
        result for request, result in acquisitions.seen if type(request).__name__ == "CountsRequest"
    ]
    assert {count.population for count in counts} == {"assigned", "triggered"}
    for count in counts:
        size = len(triggered) if count.population == "triggered" else per_arm
        assert count.counts == {"control": size, "treatment": size}
    (row,) = readout
    assert row.analysis_population == "triggered"
    assert (row.n_control, row.n_treat) == (len(triggered), len(triggered))
    assert row.mean_control == pytest.approx(statistics.fmean(expected["control"]), rel=1e-9)
    assert row.mean_treat == pytest.approx(statistics.fmean(expected["treatment"]), rel=1e-9)
    assert request.population == "triggered"


@pytest.mark.integration
def test_real_breakout_capture_checks_roster_per_declared_source():
    import ibis

    from tests.analysis_factory import view_readout
    from tests.evidence_fixtures.capture import _Capture, _wrapped_producers
    from tests.query.test_builders import _analysis_with_cross_source_breakouts

    con = ibis.duckdb.connect()
    con.create_table(
        "breakout_collision_src_a",
        obj=[
            {
                "user_id": user,
                "ts": datetime(2024, 1, 2, 9, index),
                "event": "page_view",
                "group_id": arm,
                "country_code": country,
            }
            for index, (user, arm, country) in enumerate(
                (
                    ("u1", "C", "US"),
                    ("u2", "T", "US"),
                    ("u3", "C", "CA"),
                    ("u4", "T", "CA"),
                )
            )
        ],
    )
    con.create_table(
        "breakout_collision_src_b",
        obj=[
            {
                "user_id": user,
                "ts": datetime(2024, 1, 3, 10, index),
                "event": "order",
                "order_id": f"o{index}",
                "country_code": "MX",
            }
            for index, (user, arm) in enumerate(
                (("u1", "C"), ("u2", "T"), ("u3", "C"), ("u4", "T"))
            )
        ],
    )
    experiment, resolver = _analysis_with_cross_source_breakouts(con)
    scope = Scope(view="breakout", grain="total")
    method = Method(name="unadjusted")
    declared_segments = {
        "src_a": (("US", ("C", "T")), ("CA", ("C", "T"))),
        "src_b": (("MX", ("C", "T")),),
    }
    expected = frozenset(
        CaptureKey(
            "randomized.unadjusted",
            method_identity(method),
            metric,
            scope.hash(),
            RequestCoordinates(
                "assigned",
                "total",
                segment=(
                    (f"{source}:country", value),
                    (f"{source}:country", value),
                ),
            ),
            arm,
        )
        for source, segments in declared_segments.items()
        for metric in ("visit_rate", "checkout")
        for value, arms in segments
        for arm in arms
    ) | frozenset(
        {
            CaptureKey(
                "load_breakout_evidence",
                "",
                "",
                scope.hash(),
                RequestCoordinates("assigned", "total"),
                "",
            )
        }
    )
    capture = _Capture((scope,), expected)
    try:
        with _wrapped_producers(capture):
            results = view_readout(experiment, resolver, scope)
    finally:
        resolver.close()
        con.disconnect()

    assert {row.source for row in results} == {"src_a", "src_b"}
    assert {key.request_coordinates.segment[0][0] for key, _ in capture.entries if key.metric} == {
        "src_a:country",
        "src_b:country",
    }
    assert frozenset(key for key, _ in capture.entries) == expected


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
    from typing import Any, cast

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
            envelope=cast(Any, {"coefficients": {"a": 0.25}, "residual_variance": 3.0}),
            fit_identity=("source-left", "fit"),
            evidence_sha256="left-digest",
        )
    )
    right = record(
        MethodFitAnswer(
            origin="retained_fitted",
            envelope=cast(Any, {"coefficients": {"a": 0.25}, "residual_variance": 3.0}),
            fit_identity=("source-right", "fit"),
            evidence_sha256="right-digest",
        )
    )
    assert_same_evidence(left, right)
    changed = record(
        MethodFitAnswer(
            origin="retained_fitted",
            envelope=cast(Any, {"coefficients": {"a": 0.5}, "residual_variance": 3.0}),
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


def test_source_input_bytes_preserve_original_payload_protocol():
    from decimal import Decimal

    from tests.evidence_fixtures.scenarios import _source_input_bytes, _source_plain

    payload = {
        "schema": [
            ["z", "int64"],
            ["a", "string"],
            ["flag", "boolean"],
            ["count", "int64"],
            ["value", "float64"],
            ["missing", "string"],
        ],
        "rows": [
            [2, "café", True, 9007199254740993, -0.0, None],
            [1, "east", False, 0, 1.25, None],
        ],
        "columns": ["z", "a", "flag", "count", "value", "missing"],
        "state": _source_plain(
            {
                "escape": '"\\\n',
                "decimal": Decimal("1.20"),
                "mapping": {None: "missing", 1: "one"},
            }
        ),
    }
    expected = (
        rb'{"columns":["z","a","flag","count","value","missing"],'
        rb'"rows":[[2,"caf\u00e9",true,9007199254740993,-0.0,null],'
        rb'[1,"east",false,0,1.25,null]],'
        rb'"schema":[["z","int64"],["a","string"],["flag","boolean"],'
        rb'["count","int64"],["value","float64"],["missing","string"]],'
        rb'"state":{"decimal":{"$source_decimal":"1.20"},"escape":"\"\\\n",'
        rb'"mapping":{"$source_mapping":[[1,"one"],[null,"missing"]]}}}'
    )
    assert _source_input_bytes(payload) == expected
    with pytest.raises(ValueError):
        _source_input_bytes({"value": float("nan")})


def test_source_row_receipts_preserve_empty_and_partial_batches():
    from tests.evidence_fixtures.scenarios import _source_input_bytes, _source_row_hashes

    metadata = {"schema": [["label", "string"], ["value", "float64"]]}
    batches = [
        [("café", -0.0), ('雪"\\\n', None)],
        [],
        [("last", 9007199254740993)],
    ]
    expected = (
        rb'{"rows":[["caf\u00e9",-0.0],["\u96ea\"\\\n",null],'
        rb'["last",9007199254740993]],"schema":[["label","string"],["value","float64"]]}'
    )
    assert (
        _source_input_bytes({**metadata, "rows": [row for batch in batches for row in batch]})
        == expected
    )
    digest = sha256(expected).hexdigest()
    assert _source_row_hashes(metadata, iter(batches)) == (digest, digest, 3)
    empty = rb'{"rows":[],"schema":[["label","string"],["value","float64"]]}'
    empty_digest = sha256(empty).hexdigest()
    assert _source_row_hashes(metadata, iter([[], []])) == (empty_digest, empty_digest, 0)


def test_transcript_codec_rejects_an_unregistered_version():
    from increment.evidence import CountsRequest
    from tests.evidence_fixtures.transcript import Transcript

    request = CountsRequest(kind="unit")
    transcript = Transcript(
        baseline_revision="7c324b66",
        scenario_id="scalar-checkpoint",
        ingress="from_moments",
        runtime={
            "python": "3.13.5",
            "platform": "test",
            "production_root": "/test/increment",
            "dependency_versions": {"increment": "0.0.0"},
        },
        provenance={"kind": "historical-input"},
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
    payload["schema_version"] = 3
    with pytest.raises(ValueError):
        Transcript.from_payload(payload, expected_requests=(request,))


def test_transcript_decoder_rejects_roster_and_answer_schema_drift():
    from copy import deepcopy

    from increment.evidence import CountsRequest
    from tests.evidence_fixtures.transcript import Transcript, _digest, _typed_declaration

    request = CountsRequest(kind="unit")
    transcript = Transcript(
        baseline_revision="7c324b66",
        scenario_id="scalar-checkpoint",
        ingress="from_moments",
        runtime={
            "python": "3.13.5",
            "platform": "test",
            "production_root": "/test/increment",
            "dependency_versions": {"increment": "0.0.0"},
        },
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

    def rehash(changed):
        changed["declaration_sha256"] = _digest(
            [_typed_declaration(record) for record in changed["records"]]
        )
        changed["input_sha256"] = _digest([record["answer"] for record in changed["records"]])

    missing = deepcopy(payload)
    missing["records"] = []
    rehash(missing)
    with pytest.raises(ValueError):
        Transcript.from_payload(missing, expected_requests=(request,))

    duplicate = deepcopy(payload)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    rehash(duplicate)
    with pytest.raises(ValueError):
        Transcript.from_payload(duplicate, expected_requests=(request,))

    unexpected = deepcopy(payload)
    unexpected["records"][0]["request"]["kind"] = "cluster"
    rehash(unexpected)
    with pytest.raises(ValueError):
        Transcript.from_payload(unexpected, expected_requests=(request,))

    unknown_field = deepcopy(payload)
    unknown_field["records"][0]["answer"]["discarded"] = True
    rehash(unknown_field)
    with pytest.raises(ValueError):
        Transcript.from_payload(unknown_field, expected_requests=(request,))


@pytest.mark.integration
def test_recorded_revision_must_be_a_full_commit_in_this_history():
    root = Path(__file__).parent / "fixtures" / "evidence_identity"
    recorded = json.loads((root / "scalar_checkpoint.json").read_text())["baseline_revision"]
    require_ancestor_revision(recorded)
    for revision in (recorded[:12], "f" * 40):
        with pytest.raises(ValueError):
            require_ancestor_revision(revision)


def _historical_input(name):
    import hashlib

    root = Path(__file__).parent / "fixtures" / "evidence_identity"
    payload = json.loads((root / name).read_text())
    transcript = Transcript.from_payload(
        payload,
        expected_requests=tuple(_decode_transcript_request(r) for r in payload["records"]),
        expected_scenario_id=payload["scenario_id"],
        expected_ingress=payload["ingress"],
    )
    require_ancestor_revision(transcript.baseline_revision)
    producer_pins = {
        "tests/parity_harness/cases.py": "d161e60cd18d584f067a19ba05312e86c584f296fa641e3e24af237c0a62f8a4",
        "tests/parity_harness/matrix_cases.py": "bef4af953e44d337ef9882f1762f78d65fb0a882da95500ffd022a20c518258e",
        "tests/parity_harness/matrix_data.py": "905c57b922ae7d6681efbe086158f9f718af390d9e60ec17c54fe2d0003b11bf",
        "tests/warehouse_cuped_cases.py": "082fe5f870aebb9d0ddcfd348aebac54b243430c07aeb55f7b01f2c5edcabc7e",
    }
    for item in transcript.provenance["source_inputs"]:
        assert item["path"] in producer_pins
        digest = producer_pins[item["path"]]
        assert item["sha256"] == digest
        archive = root / "producer-inputs" / f"{digest}.py.txt"
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    return transcript


@pytest.mark.integration
@pytest.mark.slow
def test_frozen_scalar_evidence_runs_mean_conversion_ratio_consumers():
    frozen = _historical_input("scalar_checkpoint.json")
    live = _acquired_coordinate(frozen.scenario_id, frozen.ingress, frozen.baseline_revision)
    for record in frozen.records:
        if record["request_type"] != "MomentsRequest":
            continue
        matches = [
            r
            for r in live.records
            if r["request_type"] == "MomentsRequest"
            and r["request"]["metric"] == record["request"]["metric"]
        ]
        assert len(matches) == 1
        _assert_input_value(matches[0]["answer"], record["answer"])
    counts = next(r["answer"] for r in frozen.records if r["request_type"] == "CountsRequest")
    assert counts == {
        "population": "assigned",
        "grain": "assignment",
        "counts": {"control": 60, "treatment": 60},
        "unit_counts": {},
    }
    assignment = next(
        r["answer"] for r in frozen.records if r["request_type"] == "AssignmentRequest"
    )
    assert assignment == {
        "experiment": "moments",
        "population": "assigned",
        "independence_grain": "unit",
        "versions": [1],
        "periods": [None],
        "arm_counts": [[1, None, "control", 60], [1, None, "treatment", 60]],
        "probabilities": [],
        "replay_status": "not_certified",
        "order_draws": [],
        "randomization_law": None,
    }


@pytest.mark.integration
def test_frozen_cuped_and_quantile_inputs_match_independent_source_rows():
    import hashlib
    import math
    from collections import Counter

    import numpy as np
    import pyarrow.parquet as pq

    from tests.estimation.test_cuped_linear import _reference
    from tests.parity_harness import matrix_data

    units = _historical_input("cuped_unit_panel_input.json")
    live_units = _acquired_coordinate(units.scenario_id, units.ingress, units.baseline_revision)
    unit_record = next(r for r in units.records if r["request_type"] == "UnitRowsRequest")
    live_record = next(
        r
        for r in live_units.records
        if r["request_type"] == "UnitRowsRequest"
        and r["request"]["metric"]["name"] == unit_record["request"]["metric"]["name"]
    )
    _assert_input_value(live_record["answer"], unit_record["answer"])
    from tests.evidence_fixtures.transcript import _restore

    rows = _restore(unit_record["answer"])["frame"]["rows"]
    raw_rows = matrix_data.linear_cuped_rows()
    raw_by_unit = {row["user_id"]: row for row in raw_rows}
    rows_by_unit = {row["unit_id"]: row for row in rows}
    assert len(rows_by_unit) == len(rows) and rows_by_unit.keys() == raw_by_unit.keys()
    assert Counter(row["group_id"] for row in rows) == {"control": 60, "treatment": 60}
    for unit_id, row in rows_by_unit.items():
        raw = raw_by_unit[unit_id]
        assert row["group_id"] == raw["variant"]
        assert row["y"] == min(11.0, max(7.0, raw["revenue"]))
        for covariate in ("country", "device", "new_user"):
            assert row[covariate] == raw[covariate]
        if raw["pre_revenue"] is None:
            assert math.isnan(row["pre_revenue"])
        else:
            assert row["pre_revenue"] == raw["pre_revenue"]

    retained = _historical_input("cuped_fit_input.json")
    live_fit = _acquired_coordinate(
        retained.scenario_id, retained.ingress, retained.baseline_revision
    )
    fit_record = next(r for r in retained.records if r["request_type"] == "MethodFitRequest")
    live_record = next(
        r
        for r in live_fit.records
        if r["request_type"] == "MethodFitRequest" and r["request"] == fit_record["request"]
    )
    _assert_input_value(live_record["answer"]["envelope"], fit_record["answer"]["envelope"])
    answer = fit_record["answer"]
    assert answer["origin"] == live_record["answer"]["origin"] == "retained_fitted"
    export_path = (
        Path(__file__).parent / "fixtures/evidence_identity/exports/mixed_warehouse.parquet"
    )
    assert hashlib.sha256(export_path.read_bytes()).hexdigest() == (
        "9c4546a1076a775665c4c580ff15eb14d21d4a43f6d97ece7e012a79bb696b1d"
    )
    old_payload = next(
        json.loads(row["cuped_evidence"])
        for row in pq.read_table(export_path).to_pylist()
        if row["cuped_evidence"] and json.loads(row["cuped_evidence"])["metric"] == "revenue"
    )
    assert old_payload["kind"] == "fitted_multicovariate"
    old_fit = old_payload["evidence"]["fit_state"]
    old_inference = old_payload["evidence"]["inference_summary"]
    fit = answer["envelope"]["fit"]
    for field, expected in {
        "adjusted_means": old_fit["adjusted_means"],
        "anchor": old_fit["anchor"],
        "arm_order": old_fit["eligible_arms"],
        "control": old_fit["control"],
        "layout": old_fit["layout"],
        "n_by_arm": old_fit["counts"],
        "p": old_fit["p"],
    }.items():
        assert fit[field] == expected
    np.testing.assert_allclose(
        fit["covariance"],
        old_inference["adjusted_component_covariance"],
        rtol=1e-12,
        atol=1e-15,
    )
    assert fit["theta"] == pytest.approx(
        [
            slope / scale
            for slope, scale in zip(old_fit["slopes"][0], old_fit["scales"], strict=True)
        ],
        rel=1e-12,
        abs=1e-15,
    )
    raw_arms = np.array([row["variant"] for row in raw_rows])
    raw_outcomes = np.array([min(11.0, max(7.0, row["revenue"])) for row in raw_rows])
    raw_covariates = np.column_stack(
        [
            [0.0 if row["pre_revenue"] is None else row["pre_revenue"] for row in raw_rows],
            [row["country"] == "south" for row in raw_rows],
            [row["country"] == "west" for row in raw_rows],
            [row["device"] == "mobile" for row in raw_rows],
            [row["new_user"] == "new" for row in raw_rows],
            [row["pre_revenue"] is None for row in raw_rows],
        ]
    )
    _, independent_se, independent_df, _, _, _ = _reference(
        raw_outcomes,
        raw_arms,
        raw_covariates,
        ("control", "treatment"),
        "treatment",
        "control",
    )
    assert fit["dof"] == pytest.approx(independent_df, rel=1e-9)
    contrast = np.array([-1.0, 1.0])
    assert contrast @ np.asarray(fit["covariance"]) @ contrast == pytest.approx(
        independent_se**2,
        rel=1e-9,
    )

    _assert_historical_quantile_inputs()


def _assert_historical_quantile_inputs():
    import datetime as dt
    from collections import Counter

    import pyarrow.compute as pc

    from tests.parity_harness.cases import _quantile_ties_event_table

    quantile = _historical_input("quantile_unit_summary_input.json")
    live_quantile = _acquired_coordinate(
        quantile.scenario_id, quantile.ingress, quantile.baseline_revision
    )
    unit_record = next(r for r in quantile.records if r["request_type"] == "UnitRowsRequest")
    live_record = next(
        r
        for r in live_quantile.records
        if r["request_type"] == "UnitRowsRequest"
        and r["request"]["metric"] == unit_record["request"]["metric"]
    )
    _assert_input_value(live_record["answer"], unit_record["answer"])
    rows = unit_record["answer"]["frame"]["rows"]
    raw_events = _quantile_ties_event_table(n_per_arm=200, family=False)
    raw_rows = (
        raw_events.filter(
            (pc.field("event") == "latency") & (pc.field("ts") >= dt.datetime(2024, 1, 2))
        )
        .select(["user_id", "group_id", "value"])
        .rename_columns(
            ["unit_id", "group_id", "y"],
        )
        .to_pylist()
    )
    observed = {(row["unit_id"], row["group_id"]): row["y"] for row in rows}
    assert len(observed) == len(rows) == 400
    assert observed == {(row["unit_id"], row["group_id"]): row["y"] for row in raw_rows}
    assert Counter(row["group_id"] for row in rows) == {"C": 200, "T": 200}


_INPUT_ROOT = Path(__file__).parent / "fixtures" / "evidence_identity" / "consumer-inputs"


def _input_name(scenario_id, ingress):
    return f"{scenario_id}--{ingress}.json.gz"


def _archived_input(coordinate):
    """Name of the protected consumer-input file a decoded transcript came from.

    Live acquisitions return None: their components were digested on this runtime, so the
    audit rehashes them without historical witnesses.
    """
    if isinstance(coordinate, Transcript):
        return _input_name(coordinate.scenario_id, coordinate.ingress)
    return None


@cache
def _acquired_coordinate(scenario_id, ingress, baseline_revision):
    """Keep materialized data only; acquisition closes every source."""
    return acquire_scenario(scenario_id, ingress, baseline_revision=baseline_revision)


@cache
def _frozen_coordinate(scenario_id, ingress):
    import gzip

    payload = json.loads(
        gzip.decompress((_INPUT_ROOT / _input_name(scenario_id, ingress)).read_bytes())
    )
    requests = tuple(_decode_transcript_request(record) for record in payload["records"])
    coordinate = Transcript.from_payload(
        payload,
        expected_requests=requests,
        expected_scenario_id=scenario_id,
        expected_ingress=ingress,
    )
    require_ancestor_revision(coordinate.baseline_revision)
    _assert_daily_consumer_inputs(coordinate)
    _audit_warehouse_input_receipts(coordinate)
    return coordinate


def _coordinate_param(entry):
    """Every declared coordinate runs in the exhaustive gate; the ceiling-scale artifact also
    carries the stress cost of its 4,000,001-unit control arm."""
    marks = [pytest.mark.slow, pytest.mark.exhaustive]
    if (entry.scenario_id, entry.ingress) == (
        "exact_binomial_beyond_the_former_arm_ceiling",
        "from_unit_day_artifact",
    ):
        marks += [pytest.mark.stress, pytest.mark.timeout(900)]
    return pytest.param(entry, id=f"{entry.scenario_id}--{entry.ingress}", marks=marks)


@pytest.mark.integration
@pytest.mark.grid("Complete declared scenario/source disposition and typed-input inventory")
@pytest.mark.parametrize("entry", [_coordinate_param(entry) for entry in scenario_roster()])
def test_declared_coordinate_acquires_exact_evidence_or_disposition(entry):
    from tests.evidence_fixtures.scenarios import _scenario_case

    expected_warnings = (
        _scenario_case(entry.scenario_id).expected_warning_codes.get(entry.ingress, ())
        if entry.family != "preflight"
        else ()
    )
    if entry.disposition == "source":
        frozen = _frozen_coordinate(entry.scenario_id, entry.ingress)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IncrementWarning)
        if entry.disposition != "source":
            result = observe_scenario_disposition(entry.scenario_id, entry.ingress)
            assert result["observed"] == result["expected"]
        else:
            live = _acquired_coordinate(entry.scenario_id, entry.ingress, frozen.baseline_revision)
    assert set(warning_codes(caught)) <= set(expected_warnings)
    if entry.disposition != "source":
        return
    assert not live.missing_representations
    assert live.declaration_sha256 == _digest(
        [[record["request_type"], record["request"]] for record in live.records]
    )
    assert live.input_sha256 == _digest([record["answer"] for record in live.records])
    modes = live.provenance["coherence"]
    assert modes["coherence"] == "captured"
    assert modes["effective_modes"]
    assert set(modes["effective_modes"].values()) == {"captured"}
    _assert_coordinate_inputs(live, frozen)


_RAW_REFERENCE_SCENARIOS = frozenset(
    {
        "audit-ratio-neighboring-means",
        "audit-cuped-covariate-rescaling",
        "exact_binomial_beyond_the_former_arm_ceiling",
    }
)
_RAW_REFERENCE_FIELDS = frozenset(
    {
        "cy1",
        "cy2",
        "cx1",
        "cx2",
        "cxy",
        "cden1",
        "cden2",
        "cden3",
        "cyden",
        "cxden",
    }
)


def _digest_input_rows(rows, *, raw=False, columns=None):
    from increment.estimation.readout_types import StreamingDigest
    from increment.readouts._source_digest import input_evidence

    digest = StreamingDigest()
    if columns is not None:
        digest.update_positional_rows(
            (tuple(row[name] for name in columns) for row in rows),
            columns=tuple(columns),
            normalize_frame_floats=True,
        )
    elif raw:
        digest.update_rows(rows, normalize_frame_floats=True)
    else:
        for row in rows:
            digest.update(input_evidence(row))
    return digest.hexdigest(), digest.count


def _audit_metric_identity(record, records):
    from tests.evidence_fixtures.transcript import _restore

    answer = _restore(record["answer"])
    request = record["request"]
    has_linear = any(
        method["variance_reduction"] == "cuped_multicovariate" for method in request["methods"]
    )
    available = {
        "moments_rows": _digest_input_rows(answer["rows"], raw=has_linear),
    }
    available.update(
        {
            f"scalar_method:{name}": _digest_input_rows(rows, raw=True)
            for name, rows in answer["method_rows"]
        }
    )
    frame = answer["unit_frame"]
    if answer["evidence_kind"] == "unit_evidence_rows" and frame is None:
        frame = next(
            _restore(item["answer"])["frame"]
            for item in records
            if item["request_type"] == "UnitRowsRequest"
            and item["request"]["metric"] == request["metric"]
            and item["request"]["population"] == request["population"]
        )
    if frame is not None:
        available["unit_evidence_rows"] = _digest_input_rows(
            frame["rows"],
            columns=frame["columns"],
        )
    for fit_request, fit_answer in answer["method_fit_answers"]:
        envelope = fit_answer["envelope"]
        if envelope is None:
            continue
        fit = envelope["fit"]
        if fit is None:
            fit_values = None
        else:
            fit_values = {
                key: fit[key]
                for key in (
                    "arm_order",
                    "control",
                    "n_by_arm",
                    "adjusted_means",
                    "theta",
                    "anchor",
                    "covariance",
                    "dof",
                    "n_min",
                    "p",
                    "max_leverage",
                    "mean_leverage",
                    "leverage_trace",
                    "is_ratio",
                )
            }
            from increment.estimation.cuped.encoding import MulticovariateCupedLayout

            fit_values["layout"] = tuple(
                fit["layout"][name] for name in MulticovariateCupedLayout.__dataclass_fields__
            )
        fit_record = {
            "fit_identity": fit_answer["fit_identity"],
            "arm_order": envelope["arm_order"],
            "withheld": tuple((arm, error["code"]) for arm, error in envelope["withheld"]),
            "fit": fit_values,
        }
        available[f"warehouse_qr:{fit_request['method']['name']}"] = _digest_input_rows(
            (fit_record,),
            raw=True,
        )
    kind = answer["evidence_kind"]
    if kind != "mixed_rows":
        assert (answer["evidence_sha256"], answer["evidence_count"]) == available[kind]
    else:
        assert answer["evidence_sha256"] == ""
    for field in ("evidence_components", "decision_evidence_components"):
        for component, digest, count in answer[field]:
            if component == "moments_rows" and not has_linear:
                expected = _digest_input_rows(answer["rows"])
            else:
                expected = available[component]
            assert (digest, count) == expected, (request, field, component)
    return answer


def _normalize_scope_identity_metadata(value, source_identity):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "source_identity":
                assert item == source_identity
                for artifact in item.get("artifacts", ()):
                    for field in ("artifact_id", "generation_id", "manifest_sha256"):
                        assert artifact[field]
                        artifact[field] = f"validated-runtime-{field}"
            elif key == "artifact_provenance":
                artifacts = source_identity.get("artifacts", ())
                assert item["manifest_sha256"] == (
                    artifacts[0]["manifest_sha256"] if artifacts else None
                )
                if artifacts:
                    item["manifest_sha256"] = "validated-runtime-manifest_sha256"
                roles = [role for role, _digest_value in item["relation_digests"]]
                for role, digest in item["relation_digests"]:
                    assert len(digest) == 64 and int(digest, 16) >= 0, role
                item["relation_digests"] = [[role, "runtime-relation-bytes"] for role in roles]
            else:
                _normalize_scope_identity_metadata(item, source_identity)
    elif isinstance(value, list):
        for item in value:
            _normalize_scope_identity_metadata(item, source_identity)


def _comparison_records(coordinate):
    import hashlib

    from tests.evidence_fixtures.transcript import _plain, _restore

    records = deepcopy(coordinate.records)
    inputs = coordinate.provenance["source_inputs"]
    source_identities = [
        item["configuration"]["source_identity"]
        for item in inputs
        if "source_identity" in item.get("configuration", {})
    ]
    retained_payloads = {
        hashlib.sha256(receipt["payload"].encode()).hexdigest(): receipt["payload"]
        for receipt in coordinate.provenance["consumer_configuration"].get("format_inputs", ())
    }

    def fit_identity(answer, request):
        if answer["origin"] == "warehouse" and answer["evidence_sha256"] is not None:
            answer["evidence_sha256"] = "validated-warehouse-relation-preimage"
        if answer["origin"] in {"retained_fitted", "retained_categorical"}:
            assert answer["evidence_sha256"] in retained_payloads
            payload = json.loads(retained_payloads[answer["evidence_sha256"]])
            assert payload["population"] == request["population"]
            assert payload["metric"] == request["metric"]["name"]
            assert payload["method"] == Method.model_validate(request["method"]).model_dump(
                mode="json",
                exclude={"name"},
            )
            answer["evidence_sha256"] = "validated-retained-payload"
        identity = answer["fit_identity"]
        if not identity:
            return
        source_mapping = dict(_restore(identity[3]))
        if "artifacts" in source_mapping:
            source_mapping["artifacts"] = [
                dict(artifact) for artifact in source_mapping["artifacts"]
            ]
        assert source_mapping == source_identities[0]
        if source_mapping.get("artifacts"):
            artifacts = source_mapping["artifacts"]
            for artifact in artifacts:
                for field in ("artifact_id", "generation_id", "manifest_sha256"):
                    assert artifact[field]
                    artifact[field] = f"runtime-{field}"
            source_mapping["artifacts"] = artifacts
        identity[3] = _plain(source_mapping)

    for original, record in zip(coordinate.records, records, strict=True):
        if record["request_type"] == "MethodFitRequest":
            fit_identity(record["answer"], record["request"])
        if record["request_type"] != "MetricEvidenceRequest":
            continue
        _audit_metric_identity(original, coordinate.records)
        answer = record["answer"]
        for request, fitted in answer["method_fit_answers"]:
            fit_identity(fitted, request)
        if answer["evidence_kind"] != "mixed_rows":
            answer["evidence_sha256"] = "validated-input-digest"
        for field in ("evidence_components", "decision_evidence_components"):
            for component in answer[field]:
                component[1] = "validated-input-digest"

    for record in records:
        _normalize_scope_identity_metadata(record["answer"], source_identities[0])
    return records


def _warehouse_answer_records(coordinate):
    from tests.evidence_fixtures.transcript import _restore

    assembly = coordinate.provenance["consumer_configuration"].get("assembly", {})
    root_identity = coordinate.provenance["source_inputs"][0]["configuration"]["source_identity"]
    if "source_identity" in assembly:
        assert assembly["source_identity"] == root_identity
    for record in coordinate.records:
        if record["request_type"] == "MethodFitRequest":
            yield record["request"], record["answer"], root_identity
        elif record["request_type"] == "MetricEvidenceRequest":
            for request, answer in _restore(record["answer"])["method_fit_answers"]:
                yield request, answer, root_identity
    for scope in assembly.get("breakout_scope_inputs", ()):
        for entry in scope["entries"]:
            record = entry["record"]
            if record["request_type"] == "MethodFitRequest":
                yield record["request"], record["answer"], scope["source_identity"]


def _warehouse_request_key(request, source_identity):
    from tests.evidence_fixtures.transcript import _plain, _restore

    return _digest(_plain((_restore(source_identity), _restore(request))))


def _warehouse_preimage_digest(receipt):
    from packaging.version import Version

    backend = receipt["backend"]
    assert backend.keys() == {"name", "module", "qualname", "version", "ibis_version"}
    assert all(isinstance(value, str) and value for value in backend.values())
    for name in _RUNTIME_VERSION_DISTRIBUTIONS:
        Version(backend[name])
    preimage = receipt["preimage"]
    assert preimage.keys() == {"protocol", "schema", "hash_sum", "salted_hash_sum", "row_count"}
    assert preimage["protocol"] == "warehouse-method-fit-input-v1"
    schema = preimage["schema"]
    assert isinstance(schema, list) and schema
    assert all(
        isinstance(column, list)
        and len(column) == 2
        and all(isinstance(value, str) and value for value in column)
        for column in schema
    )
    assert len({column[0] for column in schema}) == len(schema)
    assert all(
        isinstance(preimage[name], str) and preimage[name]
        for name in ("hash_sum", "salted_hash_sum")
    )
    assert type(preimage["row_count"]) is int and preimage["row_count"] >= 0
    payload = (
        preimage["protocol"],
        schema,
        preimage["hash_sum"],
        preimage["salted_hash_sum"],
        preimage["row_count"],
    )
    return sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _audit_warehouse_input_receipts(coordinate):
    from tests.evidence_fixtures.transcript import _plain, _restore

    assembly = coordinate.provenance["consumer_configuration"].get("assembly", {})
    receipts = assembly.get("warehouse_input_receipts", [])
    assert isinstance(receipts, list)
    available = {}
    for receipt in receipts:
        assert receipt.keys() == {"request", "source_identity", "backend", "preimage"}
        key = _warehouse_request_key(receipt["request"], receipt["source_identity"])
        assert key not in available
        _decode_transcript_request(
            {
                "request_type": "MethodFitRequest",
                "request": receipt["request"],
            }
        )
        available[key] = receipt, _warehouse_preimage_digest(receipt)
    used = set()
    for request, answer, source_identity in _warehouse_answer_records(coordinate):
        if answer["origin"] != "warehouse" or answer["evidence_sha256"] is None:
            continue
        assert source_identity is not None
        key = _warehouse_request_key(request, source_identity)
        assert key in available, (coordinate.scenario_id, coordinate.ingress, request)
        receipt, digest = available[key]
        assert answer["evidence_sha256"] == digest
        assert _plain(_restore(request)) == receipt["request"]
        identity = _restore(answer["fit_identity"])
        if identity:
            assert len(identity) in {11, 13}
            mapping = dict(identity[3])
            if "artifacts" in mapping:
                mapping["artifacts"] = [dict(artifact) for artifact in mapping["artifacts"]]
            assert _plain(mapping) == receipt["source_identity"]
        else:
            assert answer["failure"] is not None
        used.add(key)
    assert used == available.keys()
    return receipts


# Receipt fields naming an installed library release, and the distribution each names.
# DuckDB's Ibis backend reports the installed ``duckdb`` distribution as its version.
_RUNTIME_VERSION_DISTRIBUTIONS = {"version": "duckdb", "ibis_version": "ibis-framework"}


def _warehouse_receipt_projection(coordinate):
    projected = deepcopy(_audit_warehouse_input_receipts(coordinate))
    for receipt in projected:
        # Each original backend aggregate independently matches its own answer.
        # Backend fingerprints are not universal cross-runtime numeric oracles.
        for name in ("hash_sum", "salted_hash_sum"):
            receipt["preimage"][name] = "validated-own-backend-aggregate"
        # Installed releases describe the runtime, not the source input; they match
        # the recording only where that runtime is installed (_assert_runtime_versions).
        for name in _RUNTIME_VERSION_DISTRIBUTIONS:
            receipt["backend"][name] = "validated-runtime-version"
    return projected


def _assert_runtime_versions(live, frozen):
    """Each recorded release matches the live receipt wherever that release is installed."""
    from importlib.metadata import version

    def receipts(coordinate):
        assembly = coordinate.provenance["consumer_configuration"].get("assembly", {})
        return [receipt["backend"] for receipt in assembly.get("warehouse_input_receipts", [])]

    for index, (actual, recorded) in enumerate(zip(receipts(live), receipts(frozen), strict=True)):
        for name, distribution in _RUNTIME_VERSION_DISTRIBUTIONS.items():
            if version(distribution) == recorded[name]:
                assert actual[name] == recorded[name], (index, name, actual[name])


def _source_input_identity_projection(coordinate):
    provenance = coordinate.provenance
    inputs = provenance["source_inputs"]
    assert inputs
    identity = inputs[0]["configuration"]["source_identity"]
    replacements = {}
    for index, artifact in enumerate(identity.get("artifacts", ())):
        for field in ("artifact_id", "generation_id", "manifest_sha256"):
            value = artifact[field]
            assert isinstance(value, str) and value
            replacements[value] = f"runtime-artifact-{index}-{field}"
        assert artifact["experiment_id"] == identity["study_id"]
        assert len(artifact["context_sha256"]) == 64
    projection = {
        "source_type": provenance["source_type"],
        "source_key": provenance["source_key"],
        "consumer_configuration": deepcopy(provenance["consumer_configuration"]),
        "source_inputs": deepcopy(inputs),
    }
    assembly = projection["consumer_configuration"].get("assembly")
    if assembly is not None:
        if "source_identity" in assembly:
            assert assembly["source_identity"] == identity
            if identity.get("artifacts"):
                assert (
                    assembly["artifact_provenance"]["manifest_sha256"]
                    == identity["artifacts"][0]["manifest_sha256"]
                )
            relations = assembly["artifact_provenance"]["relation_digests"]
            for relation in relations:
                assert len(relation[1]) == 64 and int(relation[1], 16) >= 0
                relation[1] = "runtime-relation-bytes"
        else:
            assert assembly.keys() == {"warehouse_input_receipts"}
        if "breakout_scope_inputs" in assembly:
            assembly["breakout_scope_inputs"] = _breakout_acquisition_projection(coordinate)
        if "warehouse_input_receipts" in assembly:
            assembly["warehouse_input_receipts"] = _warehouse_receipt_projection(coordinate)
    for receipt in projection["consumer_configuration"]["format_inputs"]:
        payload = json.loads(receipt["payload"])
        if payload.get("source_sha256") is not None:
            assert len(payload["source_sha256"]) == 64 and int(payload["source_sha256"], 16) >= 0
            payload["source_sha256"] = "runtime-retained-source-bytes"
        receipt["payload"] = payload
    for index, (original, projected) in enumerate(
        zip(inputs, projection["source_inputs"], strict=True)
    ):
        config = original["configuration"]
        assert config["source_identity"] == identity
        assert config["source_declaration"] == provenance["source_key"]
        assert config["request_coordinates"] == provenance["configuration_identities"]
        assert len(original["sha256"]) == 64 and int(original["sha256"], 16) >= 0
        # Live hashes are audited from original bytes. Historical receipts lack
        # raw preimages and are not cross-process equality oracles.
        projected["sha256"] = "source-byte-provenance-receipt"
        if original["kind"] == "declared_source_table":
            projected["configuration"]["table_name"] = f"source-table-{index}"
            projected["identity"] = (
                f"{coordinate.scenario_id}:{coordinate.ingress}:source-table-{index}"
            )
        if original["kind"] == "declared_artifact_manifest":
            assert identity["artifacts"]
            artifact = identity["artifacts"][0]
            assert config["artifact_id"] == artifact["artifact_id"]
            assert config["generation_id"] == artifact["generation_id"]

    def replace_identity(value):
        if isinstance(value, str):
            for original, replacement in replacements.items():
                value = value.replace(original, replacement)
            return value
        if isinstance(value, dict):
            return {key: replace_identity(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [replace_identity(item) for item in value]
        return value

    return replace_identity(projection)


def _assert_coordinate_inputs(live, frozen):
    _assert_daily_consumer_inputs(live)
    _audit_warehouse_input_receipts(live)
    actual = _comparison_records(live)
    expected = _comparison_records(frozen)
    assert live.source_input_hashes == tuple(
        item["sha256"] for item in live.provenance["source_inputs"]
    )
    from tests.test_evidence_identity_parity import _capture

    archived_outcome = _capture()["scenarios"][live.scenario_id][live.ingress]
    if "population_identity" in archived_outcome:
        assert live.source_membership_identity is not None
        _assert_input_value(
            live.source_membership_identity,
            archived_outcome["population_identity"],
            path="source_population_membership",
        )
    _assert_input_value(
        _source_input_identity_projection(live),
        _source_input_identity_projection(frozen),
        path="source_input_identity",
    )
    _assert_runtime_versions(live, frozen)
    if live.scenario_id in _RAW_REFERENCE_SCENARIOS:
        from tests.evidence_fixtures.replay import validate_raw_moment_records

        validate_raw_moment_records(live.scenario_id, live.records)
        validate_raw_moment_records(frozen.scenario_id, frozen.records)
        for records in (actual, expected):
            for record in records:
                if record["request_type"] == "MomentsRequest":
                    collections = (record["answer"],)
                elif record["request_type"] == "MetricEvidenceRequest":
                    answer = record["answer"]
                    collections = (
                        answer["rows"],
                        *(rows for _method, rows in answer["method_rows"]),
                    )
                else:
                    continue
                for rows in collections:
                    for row in rows:
                        for field in _RAW_REFERENCE_FIELDS:
                            if row[field] is not None:
                                row[field] = "independently-validated-raw-reduction"
    _assert_input_value(actual, expected)


def _assert_input_value(actual, expected, path="typed_inputs"):
    import math
    from collections.abc import Mapping

    from increment.winsor import BootstrapRoots
    from tests.evidence_fixtures.capture import _MOMENTS
    from tests.evidence_fixtures.transcript import _canonical

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        assert actual.keys() == expected.keys(), path
        if actual.keys() == {"columns", "schema", "rows"} and {"unit_id", "group_id"} <= set(
            actual["columns"]
        ):
            _assert_input_value(actual["columns"], expected["columns"], f"{path}.columns")
            _assert_input_value(actual["schema"], expected["schema"], f"{path}.schema")
            left = {_canonical([r["unit_id"], r["group_id"]]): r for r in actual["rows"]}
            right = {_canonical([r["unit_id"], r["group_id"]]): r for r in expected["rows"]}
            assert len(left) == len(actual["rows"]) and len(right) == len(expected["rows"]), path
            _assert_input_value(left, right, f"{path}.unit_rows")
            return
        if path.rsplit(".", 1)[-1] in {"additive", "log_relative"} and actual.keys() == set(
            BootstrapRoots.model_fields
        ):
            _assert_bootstrap_series(actual, expected, path)
            return
        for key in actual:
            _assert_input_value(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        if path.endswith(".denominator_support"):
            left = {_canonical(row[0]): row for row in actual}
            right = {_canonical(row[0]): row for row in expected}
            assert len(left) == len(actual) and len(right) == len(expected), path
            _assert_input_value(left, right, path)
            return
        if actual and all(
            isinstance(row, Mapping) and {"metric", "group_id", "n", "ref_y"} <= row.keys()
            for row in (*actual, *expected)
        ):

            def identity(row):
                return _canonical({k: v for k, v in row.items() if k not in _MOMENTS})

            left = {identity(row): row for row in actual}
            right = {identity(row): row for row in expected}
            assert len(left) == len(actual) and len(right) == len(expected), path
            _assert_input_value(left, right, path)
        else:
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                _assert_input_value(left, right, f"{path}[{index}]")
    elif isinstance(actual, float) or isinstance(expected, float):
        assert not isinstance(actual, bool) and not isinstance(expected, bool), path
        if path.rsplit(".", 1)[-1] in _MOMENTS:
            from tests.test_evidence_identity_parity import _compare_aggregate

            _compare_aggregate(expected, actual)
            return
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), (
            path,
            actual,
            expected,
        )
    else:
        assert type(actual) is type(expected) and actual == expected, (path, actual, expected)


def _assert_bootstrap_series(actual, expected, path):
    """Own a bootstrap-t root series by replicate index, the way its consumers read it.

    ``point``, ``pilot_target`` and ``se`` keep the generic typed-input comparison. The
    roots keep the original parity contract (count, order, ``None`` positions, sign and
    ``TOLERANCE``) and every root must also agree on the consumer scale ``point - root * se``
    (through ``expm1`` for the log-relative series) within the consumer's interval bound.
    A pivot's error is absolute (ulps of its statistic over its replicate SE), so a relative
    rule on a nearly centred root rejects platform rounding that no consumer can observe.
    """
    from tests.test_evidence_identity_parity import _compare

    for field in ("point", "pilot_target", "se"):
        _assert_input_value(actual[field], expected[field], f"{path}.{field}")
    roots = f"{path}.roots"
    _compare(expected["roots"], actual["roots"], roots)
    relative = path.rsplit(".", 1)[-1] == "log_relative"
    for index, (left, right) in enumerate(zip(actual["roots"], expected["roots"], strict=True)):
        if left is None:
            continue
        assert _bootstrap_consumer_bound(actual, left, relative) == pytest.approx(
            _bootstrap_consumer_bound(expected, right, relative), rel=1e-12, abs=1e-14
        ), (f"{roots}[{index}]", left, right)


def _bootstrap_consumer_bound(series, root, relative):
    """The interval endpoint ``bootstrap_root_interval`` derives from one root."""
    import math

    bound = series["point"] - root * series["se"]
    if not relative:
        return bound
    try:
        return math.expm1(bound)
    except OverflowError:
        return math.inf


def _decode_covariate(value):
    if isinstance(value, str):
        return value
    from increment.semantics.models import CupedCovariateBinding

    if isinstance(value, dict) and {"property", "measure", "source"} <= value.keys():
        return CupedCovariateBinding.model_validate(value)
    from pydantic import TypeAdapter

    from increment import CupedCovariate
    from increment.semantics.models import AdjustmentCovariate

    return TypeAdapter(CupedCovariate | AdjustmentCovariate).validate_python(value)


def _decode_config(value):
    from pydantic import TypeAdapter

    from increment._analysis_config import ResolvedMetricConfig

    base = TypeAdapter(ResolvedMetricConfig).validate_python({**value, "covariates": ()})
    return replace(base, covariates=tuple(_decode_covariate(v) for v in value["covariates"]))


@cache
def _request_adapter(name):
    from pydantic import TypeAdapter

    from increment import _evidence_types
    from increment.declarations import Method
    from increment.semantics import design, models
    from increment.sequential_state import SequentialSnapshot

    adapter = TypeAdapter(_REQUEST_CLASSES[name])
    adapter.rebuild(
        _types_namespace={
            **vars(_evidence_types),
            **vars(models),
            **vars(design),
            # Request fields annotated with names imported only for type checking.
            "Method": Method,
            "SequentialSnapshot": SequentialSnapshot,
            "ExperimentDefinition": models.Experiment,
        }
    )
    return adapter


def _decode_transcript_request(record):
    request = dict(record["request"])
    if request.get("method") is not None:
        request["method"] = Method.model_validate(request["method"])
    if "methods" in request:
        request["methods"] = tuple(Method.model_validate(m) for m in request["methods"])
    replacements = {}
    if "covariates" in request:
        replacements["covariates"] = tuple(_decode_covariate(c) for c in request["covariates"])
        request["covariates"] = ()
    if request.get("configs") is not None:
        replacements["configs"] = tuple(_decode_config(c) for c in request["configs"])
        request["configs"] = ()
    if isinstance(request.get("experiment"), dict):
        from increment.semantics.models import Experiment

        request["experiment"] = Experiment.model_validate(request["experiment"])
    base = _request_adapter(record["request_type"]).validate_python(request)
    return replace(base, **replacements)


def _consumer_context(transcript):
    from increment.decision_wire import compiled_plan_from_dict
    from increment.semantics.design import Encouragement, Observational, Randomized

    configuration = transcript.provenance["consumer_configuration"]
    context = configuration["context"]
    design_types = {
        "randomized": Randomized,
        "encouragement": Encouragement,
        "observational": Observational,
    }
    design = design_types[context["design"]["mechanism"]].model_validate(context["design"])
    configs = tuple(_decode_config(c) for c in context["configs"])
    plan = compiled_plan_from_dict(context["plan"])
    return configuration, context, design, configs, plan


def _assert_current_input_scope_metadata(results, rows, receipt):
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.readout_types import (
        CellKey,
        CellRecord,
        FamilyScope,
        PopulationRoster,
        ReadoutMetadata,
        ReadoutScope,
        SamplingInference,
        SourceReadoutScope,
    )
    from tests.evidence_fixtures.transcript import _restore

    context = _decode_source_context(receipt["context"])
    request = _restore(receipt["request"])
    snapshot_id = results.metadata.scope.snapshot_id
    cells = tuple(
        sorted(
            {
                CellKey.model_validate(
                    {
                        **{name: getattr(row, name, None) for name in CellKey.model_fields},
                        "kind": "segment",
                        "analysis_population": request["analysis_population"],
                    }
                )
                for row in rows
            },
            key=lambda cell: canonical_json_bytes(cell.model_dump(mode="json")),
        )
    )
    populations = tuple(sorted({cell.analysis_population for cell in cells}))
    groups = tuple(
        sorted(
            {cell.group_id for cell in cells if cell.group_id is not None}
            | {context.design.control_group}
        )
    )
    decision_cells = tuple(cell for cell in cells if cell.method_role == "decision")
    completeness = {}
    for population in populations:
        explicit = [
            row.decision_scope_complete
            for row in rows
            if row.analysis_population == population and row.decision_scope_complete is not None
        ]
        if explicit:
            completeness[population] = all(explicit)
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=cells,
        decision_cells=decision_cells,
        rosters=tuple(
            PopulationRoster(
                analysis_population=population,
                arms=groups,
                source="observed_union",
                complete=False,
                control_group=context.design.control_group,
            )
            for population in populations
        ),
        decision_complete_by_population=completeness,
        integrity=(),
    )
    policy = context.plan.view_policies.for_view(
        "breakout",
        mechanism=context.design.mechanism,
        segmented=True,
    )
    correction = request["correction"]
    families: list[FamilyScope] = []
    if correction != "none":
        metric_groups = (
            sorted({cell.metric for cell in decision_cells})
            if correction == "bonferroni"
            else [None]
        )
        for population in populations:
            for metric in metric_groups:
                name = f"{policy.name}:{metric}" if metric is not None else policy.name
                members = tuple(
                    cell
                    for cell in decision_cells
                    if cell.analysis_population == population
                    and (metric is None or cell.metric == metric)
                )
                payload = {
                    "kind": "increment.readout.family",
                    "version": 1,
                    "source_snapshot_id": snapshot_id,
                    "analysis_population": population,
                    "view": "breakout",
                    "dimension": request["dimension"],
                    "source": request["source_name"],
                    "name": name,
                }
                families.append(
                    FamilyScope(
                        family_id="sha256:" + sha256(canonical_json_bytes(payload)).hexdigest(),
                        source_snapshot_id=snapshot_id,
                        analysis_population=population,
                        view="breakout",
                        dimension=request["dimension"],
                        source=request["source_name"],
                        name=name,
                        family={
                            "name": name,
                            "correction": correction,
                            "q": request["q"] if correction in {"bh", "e_bh"} else None,
                            "axes": ("segment",) if metric is not None else policy.axes,
                            "guarantee": "fdr" if correction in {"bh", "e_bh"} else "fwer",
                            "validity_regime": "finite_sample",
                        },
                        members=members,
                        complete=True,
                    )
                )
    ordered_families = tuple(sorted(families, key=lambda family: family.family_id))
    expected_scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=cells,
        decision_cells=decision_cells,
        populations=populations,
        families=ordered_families,
        by_source={snapshot_id: source_scope},
    )
    assert all(config.prior is None for config in context.configs)
    records: list[CellRecord] = []
    for row in results:
        cell = next(
            cell
            for cell in cells
            if cell.metric == row.metric
            and cell.group_id == row.group_id
            and cell.method == row.method
            and cell.estimand == row.estimand
            and cell.value_scale == row.value_scale
            and cell.dimension_value == row.dimension_value
        )
        assert cell.method is not None
        p_value = row.p_value() if row.sampling_available and row.lift is not None else None
        evidence = (
            PValueEvidence(cell.hypothesis(), cell.method, p_value, row.reference_kind)
            if isinstance(p_value, (int, float))
            else None
        )
        records.append(
            CellRecord(
                cell=cell,
                source_snapshot_id=snapshot_id,
                sampling=SamplingInference(
                    available=row.sampling_available,
                    evidence=evidence,
                    reason_code=row.sampling_reason_code,
                    reason_context=row.sampling_reason_context,
                ),
            )
        )
    provenance = _restore(receipt["artifact_provenance"])
    ordered_records = tuple(
        sorted(
            records,
            key=lambda record: canonical_json_bytes(record.cell.model_dump(mode="json")),
        )
    )
    expected = ReadoutMetadata(
        scope=expected_scope,
        cells=ordered_records,
        artifact_manifest_sha256=provenance["manifest_sha256"],
        artifact_relation_digests=provenance["relation_digests"],
    )
    assert results.metadata.model_dump(mode="json") == expected.model_dump(mode="json")


def _old_daily_consumer(scenario_id, ingress):
    from tests.test_evidence_identity_parity import _capture

    rows = _capture()["scenarios"][scenario_id][ingress]["daily_values"]
    owners = set()
    for row in rows:
        if row.get("assignment") == "switchback":
            owners.add("contrast")
        elif row.get("sequential_result") is not None:
            owners.add("sequential_breakout")
        elif row.get("ds") is not None:
            owners.add("day_axis")
        else:
            assert row.get("dimension") is not None
            owners.add("breakout")
    assert len(owners) <= 1
    return next(iter(owners)) if owners else None


def _assert_daily_consumer_inputs(coordinate):
    owner = _old_daily_consumer(coordinate.scenario_id, coordinate.ingress)
    required = {
        None: set(),
        "contrast": {"ContrastStatsRequest"},
        "sequential_breakout": {"SequentialCaptureRequest"},
        "day_axis": {"DayAxisRequest"},
        "breakout": {"BreakoutSlicesRequest", "BreakoutMethodMomentsRequest"},
    }[owner]
    request_types = {record["request_type"] for record in coordinate.records}
    assert required <= request_types, (coordinate.scenario_id, coordinate.ingress, owner, required)
    if "BreakoutSlicesRequest" in request_types and not request_types & {
        "DayAxisRequest",
        "SequentialCaptureRequest",
    }:
        _audit_breakout_scope_inputs(coordinate)


def _old_coordinate_rows(scenario_id, ingress):
    from tests.test_evidence_identity_parity import _capture

    outcome = _capture()["scenarios"][scenario_id][ingress]
    return [
        cell["result_cell"]["retained_row"]
        for cell in outcome["cells"].values()
        if cell["result_cell"]["retained_row"] is not None
    ]


def _result_coordinate(row):
    return tuple(
        row.get(field) for field in ("metric", "group_id", "method", "estimand", "value_scale")
    )


@cache
def _scalar_consumer(encoded):
    from pydantic import TypeAdapter

    from increment.estimation.engine import _estimate_lift
    from increment.semantics.models import Metric
    from tests.evidence_fixtures.transcript import _restore

    arguments = _restore(json.loads(encoded))
    metric = TypeAdapter(Metric).validate_python(arguments.pop("metric"))
    arguments["methods"] = [Method.model_validate(method) for method in arguments["methods"]]
    prior = arguments.pop("prior")
    if prior is not None:
        from increment.estimation.inference import Prior

        prior = TypeAdapter(Prior).validate_python(prior)
    return _estimate_lift([metric], prior=prior, **arguments)


_CONSUMED_RESULTS = {}


def _consume_once(kind, data, configuration, compute):
    from tests.evidence_fixtures.transcript import _plain

    key = kind, _digest(_plain({"data": data, "configuration": configuration}))
    if key not in _CONSUMED_RESULTS:
        _CONSUMED_RESULTS[key] = compute()
    return _CONSUMED_RESULTS[key]


_FIXED_COMPUTATIONS = {}
_FIXED_PUBLISHED_ROWS = {}
_OWNED_READOUT_IDENTITIES = {}
_ASSERTED_SNAPSHOT_IDS = set()
_OLD_METADATA_SCOPES = {}
_SNAPSHOT_INPUT_SOURCES = {}
_ASSERTED_FAMILY_IDS = set()
_METADATA_ARTIFACT_INPUTS = {}


def _assert_snapshot_preimage(results, preimage):
    expected = "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest()
    assert results.metadata.scope.snapshot_id == expected
    assert {row.source_snapshot_id for row in results} <= {expected}
    _ASSERTED_SNAPSHOT_IDS.add(expected)
    _SNAPSHOT_INPUT_SOURCES[expected] = deepcopy(preimage["source"])


def _register_readout_identities(results, *, artifact_provenance):
    from datetime import date
    from hashlib import sha256

    from increment.estimation.readout_types import CellKey, serialize_day_axis
    from tests.evidence_fixtures.transcript import _restore

    metadata = results.metadata
    assert metadata is not None and metadata.scope is not None
    provenance: dict[str, Any] = (
        {"manifest_sha256": None, "relation_digests": []}
        if artifact_provenance is None
        else _restore(artifact_provenance)
    )
    assert metadata.artifact_manifest_sha256 == provenance["manifest_sha256"]
    assert [list(pair) for pair in metadata.artifact_relation_digests] == [
        list(pair) for pair in provenance["relation_digests"]
    ]
    _METADATA_ARTIFACT_INPUTS[metadata.scope.snapshot_id] = provenance
    family_ids = [family.family_id for family in metadata.scope.families]
    assert family_ids == sorted(set(family_ids))
    for family in metadata.scope.families:
        look = family.identity_look
        payload = {
            "kind": "increment.readout.family",
            "version": 2 if family.identity_metric is not None or look is not None else 1,
            "source_snapshot_id": family.decision_snapshot_id or family.source_snapshot_id,
            "analysis_population": family.analysis_population,
            "view": family.view,
            "dimension": family.dimension,
            "source": family.source,
            "name": family.name,
        }
        if family.identity_metric is not None:
            payload["metric"] = family.identity_metric
        if look is not None:
            serialized = serialize_day_axis(look)
            payload["look"] = (
                serialized
                if isinstance(look, date) and not isinstance(look, datetime)
                else {"type": type(look).__name__, "value": serialized}
            )
        assert family.family_id == "sha256:" + sha256(canonical_json_bytes(payload)).hexdigest()
        assert family.source_snapshot_id in metadata.scope.by_source
        _ASSERTED_FAMILY_IDS.add(family.family_id)
    for row in results:
        source_id = row.source_snapshot_id
        cell = CellKey.from_row(row)
        assert source_id in metadata.scope.by_source
        assert cell in metadata.scope.by_source[source_id].cells
        assert any(
            record.source_snapshot_id == source_id and record.cell == cell
            for record in metadata.cells
        )
        families = [
            family
            for family in metadata.scope.families
            if family.source_snapshot_id == source_id and cell in family.members
        ]
        family_id = getattr(row, "family_id", None)
        assert [family.family_id for family in families] == (
            [] if family_id is None else [family_id]
        )
        identity = source_id, _full_readout_coordinate(row.model_dump(mode="json"))
        assert (
            identity not in _OWNED_READOUT_IDENTITIES
            or _OWNED_READOUT_IDENTITIES[identity] == family_id
        )
        _OWNED_READOUT_IDENTITIES[identity] = family_id


def _input_evidence_projection(value, *, field=None):
    if isinstance(value, float):
        return (
            value if field in {"dimension_value", "ds", "n_control", "n_treat", "segment"} else None
        )
    if isinstance(value, int) and not -(2**53) < value < 2**53:
        return {"$type": "integer", "value": str(value)}
    if isinstance(value, dict):
        excluded = {
            "exact_delta_total",
            "discovery",
            "multiplicity_status",
            "family_id",
            "family_size",
            "family_axes",
            "family_q",
            "family_threshold",
            "family_procedure",
            "family_nominal_alpha",
            "decision_scope_complete",
            "decision_scope_reason_code",
            "decision_scope_reason_context",
            "sampling_available",
            "sampling_reason_code",
            "sampling_reason_context",
            "failure_code",
            "failure_context",
            "inference",
            "metric_type",
            "mean_control",
            "mean_treat",
        }
        return {
            key: _input_evidence_projection(item, field=key)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, (tuple, list)):
        return [_input_evidence_projection(item, field=field) for item in value]
    return value


def _snapshot_request(value, *, field="request"):
    if callable(value):
        from importlib import import_module

        module, qualname = value.__module__, value.__qualname__
        assert module and qualname and "<lambda>" not in qualname and "<locals>" not in qualname, (
            field
        )
        resolved = import_module(module)
        for name in qualname.split("."):
            resolved = getattr(resolved, name)
        assert resolved is value, field
        return {"kind": "callable", "module": module, "qualname": qualname}
    if isinstance(value, dict):
        is_method = {"name", "variance_reduction"} <= value.keys()
        return {
            key: _snapshot_request(item, field=f"{field}.{key}")
            for key, item in value.items()
            if not is_method or (key != "procedure" and (key != "covariates" or item is not None))
        }
    if isinstance(value, (tuple, list)):
        return [_snapshot_request(item, field=field) for item in value]
    return value


def _snapshot_config(config, design):
    methods = (
        ()
        if config.methods_explicitly_empty
        else (
            Method(name="iptw")
            if config.decision_defaulted and design.mechanism == "observational"
            else config.decision_method,
            *config.sensitivity_methods,
        )
    )
    assert len({method.name for method in methods}) == len(methods)
    result = {
        "metric": config.metric.model_dump(mode="json"),
        "methods": [_snapshot_request(method.model_dump(mode="python")) for method in methods],
        "prior": None if config.prior is None else config.prior.model_dump(mode="json"),
        "prior_is_global": config.prior_is_global,
    }
    if config.covariates:
        result["covariates"] = [
            _snapshot_request(covariate.model_dump(mode="json")) for covariate in config.covariates
        ]
    return result


def _composite_input_source(components):
    ordered = sorted(
        components,
        key=lambda item: (
            item["kind"],
            (item["metric"] is not None, item["metric"] or ""),
            item["population"],
            (item["source"] is not None, item["source"] or ""),
            (item["dimension"] is not None, item["dimension"] or ""),
        ),
    )
    return {
        "kind": "composite",
        "components": ordered,
        "sha256": sha256(canonical_json_bytes(ordered)).hexdigest(),
    }


def _observational_input_identity(transcript, selected, configs, design, rows, population):
    import math

    import pandas as pd

    from tests.evidence_fixtures.transcript import _restore

    configured = {
        (config.metric.name, method["name"])
        for config in configs
        for method in _snapshot_config(config, design)["methods"]
        if method["name"] in {"iptw", "dml", "aipw"}
    }
    used = {
        row.metric
        for row in rows
        if row.failure_code is None and (row.metric, row.method) in configured
    }
    identity = {}
    for metric in selected:
        if metric.name not in used:
            continue
        records = [
            record
            for record in transcript.records
            if record["request_type"] == "UnitRowsRequest"
            and record["request"]["metric"] == metric.model_dump(mode="json")
            and record["request"]["population"] == population
        ]
        assert len(records) == 1
        assert tuple(records[0]["request"]["covariates"]) == design.adjustment.covariates
        wire = _restore(records[0]["answer"])["frame"]
        frame = pd.DataFrame(wire["rows"], columns=wire["columns"])
        consumed = sorted(
            frame.to_dict(orient="records"),
            key=lambda row: (str(row.get("group_id")), str(row.get("unit_id"))),
        )
        normalized = (
            {
                key: ("nan" if math.isnan(value) else "inf" if value > 0 else "-inf")
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for key, value in row.items()
            }
            for row in consumed
        )
        digest, count = _independent_multiset_digest(normalized)
        identity[metric.name] = {
            "columns": tuple(frame.columns),
            "sha256": digest,
            "n_rows": count,
        }
    return identity or None


def _materialized_rows_digest(rows, *, raw=False, columns=None):
    import math

    from increment.estimation.cuped.encoding import categorical_missing_wire_value

    def normalize(value):
        missing = categorical_missing_wire_value(value)
        if missing is not None:
            return missing
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        return value

    def normalize_row(row):
        selected = row if columns is None else {name: row[name] for name in columns}
        if raw or columns is not None:
            selected = {
                name: (
                    None if math.isnan(value) else {"__float__": "+inf" if value > 0 else "-inf"}
                )
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for name, value in selected.items()
            }
        return normalize(
            selected if raw or columns is not None else _input_evidence_projection(selected)
        )

    return _independent_multiset_digest(normalize_row(row) for row in rows)


def _independent_multiset_digest(rows):
    accumulator = 0
    count = 0
    for row in rows:
        accumulator += int.from_bytes(sha256(canonical_digest_bytes(row)).digest(), "big")
        count += 1
    payload = {
        "kind": "increment.readout.rows",
        "version": 1,
        "count": count,
        "accumulator": f"{accumulator % (1 << 256):064x}",
    }
    return sha256(canonical_json_bytes(payload)).hexdigest(), count


def _breakout_fit_input_record(answer, request):
    from tests.evidence_fixtures.replay import _coded_error, _fitted_cuped_envelope
    from tests.evidence_fixtures.transcript import _restore

    identity = _restore(answer["fit_identity"])
    if answer["failure"] is not None:
        return {
            "fit_identity": identity,
            "failure": _coded_error(answer["failure"]).code,
            "fit": None,
        }
    envelope = _fitted_cuped_envelope(answer, request)
    withheld = tuple(
        (arm, refusal.code, dict(refusal.context)) for arm, refusal in envelope.withheld
    )
    fit = envelope.fit
    if fit is None:
        return {"fit_identity": identity, "withheld": withheld, "fit": None}
    values = {
        name: getattr(fit, name)
        for name in (
            "arm_order",
            "control",
            "n_by_arm",
            "adjusted_means",
            "theta",
            "anchor",
            "covariance",
            "dof",
            "n_min",
            "p",
            "max_leverage",
            "mean_leverage",
            "leverage_trace",
            "is_ratio",
        )
    }
    values["layout"] = tuple(getattr(fit.layout, name) for name in fit.layout.__dataclass_fields__)
    projections = []
    for arm in fit.arm_order:
        if arm == fit.control:
            continue
        try:
            projection = (
                fit.ratio(arm, control=fit.control)
                if fit.is_ratio
                else fit.relative(arm, control=fit.control)
            )
        except CodedError as error:
            projections.append((arm, error.code, dict(error.context)))
        else:
            projections.append(
                (
                    arm,
                    projection.estimate,
                    projection.se,
                    projection.dof,
                    projection.lower,
                    projection.upper,
                )
            )
    return {
        "fit_identity": identity,
        "withheld": withheld,
        "fit": values,
        "projections": tuple(projections),
    }


_WITNESSES = Path(__file__).parent / "evidence_fixtures" / "breakout_component_witnesses.json"
_WITNESS_COORDINATE = ("fixture", "receipt", "entry", "kind", "metric", "arm", "alpha")
_PROJECTION_ALPHA = 0.05


@cache
def _breakout_component_witnesses():
    """Historical interval bounds inside archived warehouse fit-record component preimages.

    A fit-record component digests the fit's stored values plus each treatment arm's
    projection, whose ``lower``/``upper`` are recomputed through the Student-t inverse:
    the archived bytes are the capture platform's rounding, not decoded data. A witness
    carries exactly those bytes for one coordinate and has no authority of its own: the
    archived record rehashed with the witnessed bounds must equal the protected component
    digest, and the locally recomputed bounds must agree with the witnessed ones at the
    original parity contract.
    """
    witnesses = {}
    for witness in json.loads(_WITNESSES.read_text()):
        assert witness.keys() == {*_WITNESS_COORDINATE, "archived_sha256", "lower", "upper"}
        key = tuple(witness[name] for name in _WITNESS_COORDINATE)
        assert key not in witnesses, key
        witnesses[key] = witness
    return witnesses


def _witnessed_fit_input_record(record, archived, kind, metric, component):
    """Rebind an archived fit record's recomputed interval bounds to their witnessed bytes."""
    from tests.test_evidence_identity_parity import _compare

    if "projections" not in record:
        return record
    fixture, receipt_index, entry_index = archived
    projections = []
    for projection in record["projections"]:
        if len(projection) != 6:
            projections.append(projection)
            continue
        arm, estimate, se, dof, lower, upper = projection
        key = (fixture, receipt_index, entry_index, kind, metric, arm, _PROJECTION_ALPHA)
        witness = _breakout_component_witnesses().get(key)
        assert witness is not None, key
        assert witness["archived_sha256"] == component["sha256"], key
        label = f"{fixture}[{receipt_index}][{entry_index}].{arm}"
        _compare(witness["lower"], lower, f"{label}.lower")
        _compare(witness["upper"], upper, f"{label}.upper")
        projections.append((arm, estimate, se, dof, witness["lower"], witness["upper"]))
    return {**record, "projections": tuple(projections)}


@pytest.mark.integration
def test_breakout_component_witnesses_cover_archived_fit_projections_exactly():
    import gzip

    expected = {}
    for path in sorted(_INPUT_ROOT.glob("*.json.gz")):
        payload = json.loads(gzip.decompress(path.read_bytes()))
        assembly = payload["provenance"]["consumer_configuration"].get("assembly", {})
        for receipt_index, receipt in enumerate(assembly.get("breakout_scope_inputs", ())):
            for entry_index, entry in enumerate(receipt["entries"]):
                component = entry["component"]
                if component is None or not component["kind"].startswith("warehouse_qr_breakout:"):
                    continue
                record = entry["record"]
                fit_record = _breakout_fit_input_record(
                    record["answer"], _decode_transcript_request(record)
                )
                for projection in fit_record.get("projections", ()):
                    if len(projection) == 6:
                        arm = projection[0]
                        key = (path.name, receipt_index, entry_index, component["kind"])
                        expected[(*key, component["metric"], arm, _PROJECTION_ALPHA)] = component[
                            "sha256"
                        ]
    witnesses = _breakout_component_witnesses()
    assert {key: witness["archived_sha256"] for key, witness in witnesses.items()} == expected


def _breakout_frame_component(record, config, receipt):
    from tests.evidence_fixtures.transcript import _restore

    request = _decode_transcript_request(record)
    answer = _restore(record["answer"])
    frame = answer["frame"]
    rows = [dict(row) for row in frame["rows"]]
    columns = list(frame["columns"])
    assert len(columns) == len(set(columns))
    assert all(set(row) == set(columns) for row in rows)
    names = tuple(covariate.name for covariate in config.covariates)
    assert request.covariates == names
    dimension = receipt["identity"]["property"]
    if answer["unit_frame_has_dimension"]:
        requested = tuple(dict.fromkeys((dimension, *names)))
        covariate_columns = {}
        for name in requested:
            column = name
            if name in {"unit_id", "group_id", "y", "y_den"}:
                column = f"__covariate_{name}"
                while column in requested or column in covariate_columns.values():
                    column += "_"
            covariate_columns[name] = column
        alias = "__breakout_dimension"
        while alias in names:
            alias += "_"
        if alias not in columns:
            columns.append(alias)
        for row in rows:
            row[alias] = row[covariate_columns[dimension]]
    digest, count = _materialized_rows_digest(rows, columns=columns)
    return {
        "kind": "unit_evidence_rows",
        "metric": request.metric.name,
        "population": request.population,
        "source": receipt["request"]["source_name"],
        "dimension": dimension,
        "sha256": digest,
        "count": count,
    }


def _assert_breakout_fit_association(answer, request, receipt):
    from tests.evidence_fixtures.transcript import _plain, _restore

    assert request.cell is not None
    identity = _restore(answer["fit_identity"])
    if not identity:
        assert answer["failure"] is not None
        return
    assert len(identity) in {11, 13}
    assert tuple(identity[:3]) == (
        "warehouse-cuped-fit-v1",
        request.metric.name,
        request.population,
    )
    mapping = dict(_restore(identity[3]))
    if "artifacts" in mapping:
        mapping["artifacts"] = [dict(artifact) for artifact in mapping["artifacts"]]
    assert mapping == _restore(receipt["source_identity"])
    assert (
        identity[4]
        == canonical_json_bytes(request.method.model_dump(mode="json", exclude={"name"})).decode()
    )
    assert identity[5] == request.control and tuple(identity[6]) == request.arm_order
    cell = request.cell
    assert _plain(identity[7:11]) == _plain(
        (
            cell.breakout.property,
            cell.breakout.source,
            cell.value,
            cell.is_null,
        )
    )
    envelope = answer["envelope"]
    if envelope is not None and envelope["fit"] is not None:
        fit_data = envelope["fit"]
        if "code" not in fit_data:
            assert "_projections" in fit_data
            keys = [tuple(projection[:4]) for projection in fit_data["_projections"]]
            assert len(keys) == len(set(keys))


def _breakout_acquisition_projection(coordinate):
    validated = _audit_breakout_scope_inputs(coordinate)
    projected = deepcopy([receipt for receipt, _components in validated])
    for scope_index, receipt in enumerate(projected):
        replacements = {}
        for index, artifact in enumerate(receipt["source_identity"]["artifacts"]):
            for name in ("artifact_id", "generation_id", "manifest_sha256"):
                replacements[artifact[name]] = f"scope-{scope_index}-artifact-{index}-{name}"
        provenance = receipt["artifact_provenance"]
        provenance["relation_digests"] = [
            [role, "validated-runtime-relation-bytes"]
            for role, _digest_value in provenance["relation_digests"]
        ]
        for entry in receipt["entries"]:
            if entry["component"] is not None:
                entry["component"]["sha256"] = "validated-component-preimage"
            record = entry["record"]
            if record["request_type"] == "BreakoutUnitRowsRequest":
                frame = record["answer"]["frame"]
                keys = [(row["unit_id"], row["group_id"]) for row in frame["rows"]]
                assert len(keys) == len(set(keys))
                frame["rows"] = sorted(
                    frame["rows"],
                    key=lambda row: canonical_digest_bytes((row["unit_id"], row["group_id"])),
                )
            elif record["answer"]["evidence_sha256"] is not None:
                if record["answer"]["origin"] == "warehouse":
                    record["answer"]["evidence_sha256"] = "validated-warehouse-relation-preimage"
                else:
                    assert record["answer"]["origin"] in {"retained_fitted", "retained_categorical"}
                    record["answer"]["evidence_sha256"] = "validated-original-fitted-payload"
        payloads = []
        for payload in receipt["categorical_payloads"]:
            value = json.loads(payload)
            if value.get("source_sha256") is not None:
                assert len(value["source_sha256"]) == 64 and int(value["source_sha256"], 16) >= 0
                value["source_sha256"] = "runtime-retained-source-bytes"
            payloads.append(value)
        receipt["categorical_payloads"] = payloads

        def replace_artifacts(value, replacements=replacements):
            if isinstance(value, str):
                for original, replacement in replacements.items():
                    value = value.replace(original, replacement)
                return value
            if isinstance(value, dict):
                return {name: replace_artifacts(item) for name, item in value.items()}
            if isinstance(value, list):
                return [replace_artifacts(item) for item in value]
            return value

        projected[scope_index] = replace_artifacts(receipt)
    return projected


def _assert_breakout_receipt_domain(receipt, coordinate, configs):
    from tests.evidence_fixtures.transcript import _restore

    local_metrics = {
        entry["record"]["request"]["metric"]["name"]
        for entry in receipt["entries"]
        if entry["usage"] == "local_breakout_unit_frame"
    }
    actual_fits = [
        _decode_transcript_request(entry["record"])
        for entry in receipt["entries"]
        if entry["record"]["request_type"] == "MethodFitRequest"
    ]
    for config in configs:
        methods = [
            method
            for method in [config.decision_method, *config.sensitivity_methods]
            if method.variance_reduction == "cuped_multicovariate"
        ]
        if not methods:
            continue
        metric_fits = [request for request in actual_fits if request.metric == config.metric]
        if not metric_fits:
            assert config.metric.name in local_metrics
            continue
        for method in methods:
            records = [
                record
                for record in coordinate.records
                if record["request_type"] == "BreakoutMethodMomentsRequest"
                and record["request"]["metric"] == config.metric.model_dump(mode="json")
                and record["request"]["method"] == method.model_dump(mode="json")
                and record["request"]["breakout"] == receipt["identity"]
                and record["request"]["population"] == receipt["request"]["analysis_population"]
            ]
            assert len(records) == 1
            rows = _restore(records[0]["answer"]["rows"])
            groups = {}
            for row in rows:
                # These closed fitted inputs have all arms admitted to their fit cells.
                assert int(row["n"]) >= 2
                assert row["ref_y"] + row["cy1"] / row["n"] > 0.0
                segment = (
                    "__null__"
                    if row[receipt["identity"]["property"]] is None
                    else str(row[receipt["identity"]["property"]])
                )
                groups.setdefault(segment, set()).add(str(row["group_id"]))
            observed = [request for request in metric_fits if request.method == method]
            assert len(observed) == len(groups)
            assert {
                ("__null__" if request.cell.is_null else str(request.cell.value))
                for request in observed
            } == groups.keys()
            for request in observed:
                segment = "__null__" if request.cell.is_null else str(request.cell.value)
                arms = groups[segment]
                assert request.control in arms
                assert request.arm_order == (request.control, *sorted(arms - {request.control}))


def _breakout_scope_request(receipt, configuration, context):
    from tests.evidence_fixtures.transcript import _plain, _restore

    request = _restore(receipt["request"])
    assert request.keys() == {
        "source_identity",
        "metrics",
        "configs",
        "dimension",
        "estimands",
        "correction",
        "q",
        "source_name",
        "analysis_population",
    }
    assert _plain(request["source_identity"]) == receipt["source_identity"]
    assert request["dimension"] == receipt["identity"]["property"]
    source_name = receipt["stamp_source"]
    if source_name is None:
        source_name = receipt["source_name"]
    assert request["source_name"] == source_name
    assert request["analysis_population"] == configuration["population"]
    assert _plain(request["estimands"]) == configuration["estimands"]
    _assert_input_value(
        request["configs"],
        [
            _snapshot_config(config, context.design)
            for config in sorted(context.configs, key=lambda value: value.metric.name)
        ],
    )
    assert request["metrics"] == [
        config.metric.model_dump(mode="json")
        for config in sorted(context.configs, key=lambda value: value.metric.name)
    ]
    policy = context.plan.view_policies.for_view(
        "breakout",
        mechanism=context.design.mechanism,
        segmented=True,
    )
    assert request["correction"] == ("bh" if policy.correction == "e_bh" else policy.correction)
    assert request["q"] == (policy.q if policy.q is not None else context.plan.q)
    return request


def _breakout_input_component(entry, typed, config, receipt, context, *, archived=None):
    record = entry["record"]
    identity = receipt["identity"]
    source_name = receipt["request"]["source_name"]
    if entry["usage"] == "local_breakout_unit_frame":
        assert record["request_type"] == "BreakoutUnitRowsRequest"
        assert typed.breakout.property == identity["property"]
        assert typed.breakout.source == source_name
        return _breakout_frame_component(record, config, receipt)
    assert record["request_type"] == "MethodFitRequest"
    assert typed.cell is not None
    assert typed.cell.breakout.property == identity["property"]
    assert typed.cell.breakout.source == source_name
    assert typed.control == context.design.control_group
    assert typed.method in [config.decision_method, *config.sensitivity_methods]
    assert typed.method.variance_reduction == "cuped_multicovariate"
    answer = record["answer"]
    if entry["usage"] == "unavailable_breakout_fit":
        assert answer["origin"] == "unavailable"
        return None
    _assert_breakout_fit_association(answer, typed, receipt)
    segment = "__null__" if typed.cell.is_null else str(typed.cell.value)
    if entry["usage"] == "warehouse_breakout_fit":
        assert answer["origin"] in {"retained_fitted", "warehouse"}
        kind = f"warehouse_qr_breakout:{typed.method.name}:{identity['property']}:{segment}"
        fit_record = _breakout_fit_input_record(answer, typed)
        if archived is not None:
            fit_record = _witnessed_fit_input_record(
                fit_record, archived, kind, typed.metric.name, entry["component"]
            )
        digest, count = _materialized_rows_digest([fit_record], raw=True)
    else:
        assert entry["usage"] == "retained_categorical_breakout_fit"
        assert answer["origin"] == "retained_categorical" and answer["failure"] is None
        matches = [
            payload
            for payload in receipt["categorical_payloads"]
            if sha256(payload.encode()).hexdigest() == answer["evidence_sha256"]
        ]
        assert len(matches) == 1
        payload = json.loads(matches[0])
        assert payload["metric"] == typed.metric.name and payload["population"] == typed.population
        assert payload["method"] == typed.method.model_dump(mode="json", exclude={"name"})
        digest, count = answer["evidence_sha256"], 1
        kind = f"categorical_cells_breakout:{typed.method.name}:{identity['property']}:{segment}"
    return {
        "kind": kind,
        "metric": typed.metric.name,
        "population": typed.population,
        "source": source_name,
        "dimension": identity["property"],
        "sha256": digest,
        "count": count,
    }


def _assert_breakout_slice_association(receipt, coordinate):
    identity = receipt["identity"]
    roots = [
        record
        for record in coordinate.records
        if record["request_type"] == "BreakoutSlicesRequest"
        and identity in record["request"]["breakouts"]
    ]
    assert len(roots) == 1
    root = roots[0]
    if receipt["input_kind"] == "materialized_breakout_slice":
        matches = [item for item in root["answer"]["slices"] if item["identity"] == identity]
        assert len(matches) == 1
        assert matches[0]["stamp_source"] == receipt["stamp_source"]
        assert matches[0]["source_name"] == receipt["source_name"]
    else:
        assert receipt["input_kind"] == "summary_breakout_source_slice"
        assert root["answer"]["route"] == "unsupported" and root["answer"]["slices"] == []
    assert receipt["route"] in {"native", "moments", "artifact"}


def _audit_breakout_scope_inputs(coordinate):
    from tests.evidence_fixtures.transcript import _validate_record

    archive = _archived_input(coordinate)
    configuration = coordinate.provenance["consumer_configuration"]
    assembly = configuration.get("assembly", {})
    receipts: list[Any] = assembly.get("breakout_scope_inputs", [])
    assert isinstance(receipts, list)
    scope_keys = {
        "identity",
        "source_name",
        "stamp_source",
        "context",
        "source_identity",
        "artifact_provenance",
        "request",
        "entries",
        "categorical_payloads",
        "route",
        "input_kind",
    }
    seen_scopes = set()
    expected_scopes = [
        (identity["property"], identity["source"], source)
        for record in coordinate.records
        if record["request_type"] == "BreakoutSlicesRequest"
        for identity, source in _breakout_source_partitions(record)
    ]
    ordered_scopes = []
    validated = []
    for receipt_index, receipt in enumerate(receipts):
        assert receipt.keys() == scope_keys
        identity = receipt["identity"]
        assert identity.keys() == {"property", "source"}
        scope_key = identity["property"], identity["source"], receipt["stamp_source"]
        ordered_scopes.append(scope_key)
        assert scope_key not in seen_scopes
        seen_scopes.add(scope_key)
        _assert_breakout_slice_association(receipt, coordinate)
        context = _decode_source_context(receipt["context"])
        assert context.study_id == receipt["source_identity"]["study_id"]
        assert (
            context.study_id
            == coordinate.provenance["source_inputs"][0]["configuration"]["source_identity"][
                "study_id"
            ]
        )
        provenance = receipt["artifact_provenance"]
        assert provenance.keys() == {"manifest_sha256", "relation_digests"}
        artifacts = receipt["source_identity"]["artifacts"]
        assert provenance["manifest_sha256"] == (
            artifacts[0]["manifest_sha256"] if artifacts else None
        )
        for role, digest in provenance["relation_digests"]:
            assert isinstance(role, str) and len(digest) == 64 and int(digest, 16) >= 0
        configs = tuple(context.configs)
        by_metric = {config.metric.name: config for config in configs}
        assert len(by_metric) == len(configs)
        request = _breakout_scope_request(receipt, configuration, context)
        seen_requests = set()
        covered = set()
        components = []
        for entry_index, entry in enumerate(receipt["entries"]):
            assert entry.keys() == {"usage", "record", "component"}
            record = entry["record"]
            _validate_record(record)
            typed = _decode_transcript_request(record)
            record_key = record["request_type"], _digest(record["request"])
            assert record_key not in seen_requests
            seen_requests.add(record_key)
            metric = typed.metric.name
            assert metric in by_metric
            config = by_metric[metric]
            assert typed.metric == config.metric
            assert typed.population == request["analysis_population"]
            expected_component = _breakout_input_component(
                entry,
                typed,
                config,
                receipt,
                context,
                archived=None if archive is None else (archive, receipt_index, entry_index),
            )
            assert entry["component"] == expected_component
            covered.add(metric)
            if expected_component is not None:
                components.append(expected_component)
        qualifying = {
            config.metric.name
            for config in configs
            if any(
                method.variance_reduction == "cuped_multicovariate"
                for method in [config.decision_method, *config.sensitivity_methods]
            )
        }
        assert covered == qualifying
        _assert_breakout_receipt_domain(receipt, coordinate, configs)
        validated.append((receipt, components))
    assert ordered_scopes == expected_scopes
    return validated


def _design_input_source(
    transcript, population, design, evidence, summary, roster, *, decision=False
):
    from tests.evidence_fixtures.transcript import _restore

    components = []
    for metric in sorted(evidence):
        value = evidence[metric]
        if design.mechanism == "encouragement":
            matching = [
                record
                for record in transcript.records
                if record["request_type"] == "MetricEvidenceRequest"
                and record["request"]["metric"]["name"] == metric
                and record["request"]["population"] == population
            ]
            if matching:
                assert len(matching) == 1
                answer = _audit_metric_identity(matching[0], transcript.records)
                component_key = (
                    "decision_evidence_components" if decision else "evidence_components"
                )
                parts = [(kind, digest) for kind, digest, _count in answer[component_key]]
            else:
                moment_records = [
                    record
                    for record in transcript.records
                    if record["request_type"] == "MomentsRequest"
                    and record["request"]["metric"]["name"] == metric
                    and record["request"]["population"] == population
                ]
                assert len(moment_records) == 1
                digest, count = _materialized_rows_digest(_restore(moment_records[0]["answer"]))
                assert value.evidence_components == (("moments_rows", digest, count),)
                parts = [("moments_rows", digest)]
        else:
            digest, count = _materialized_rows_digest(value.rows)
            assert count == len(value.rows)
            parts = [("moments_rows", digest)]
        components.extend(
            {
                "kind": kind,
                "metric": metric,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": digest,
            }
            for kind, digest in parts
        )
    if summary is not None:
        payload = {
            name: getattr(summary, name)
            for name in (
                "study_id",
                "cohort",
                "window_days",
                "one_sided",
                "cluster",
                "control_group",
            )
        }
        payload["as_of"] = None if summary.as_of is None else str(summary.as_of)
        payload["arms"] = [
            {name: getattr(arm, name) for name in arm.__dataclass_fields__}
            for arm in sorted(summary.arms, key=lambda arm: arm.group_id)
        ]
        components.append(
            {
                "kind": "compliance_summary",
                "metric": None,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
            }
        )
    if roster is not None and roster["counts"] is not None:
        components.append(
            {
                "kind": "assignment_counts",
                "metric": None,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": sha256(canonical_digest_bytes(roster["counts"])).hexdigest(),
            }
        )
    return _composite_input_source(components)


def _sequential_scope_with_inputs(record, declarations, options):
    from tests.evidence_fixtures.temporal import (
        scope_sequential_breakout_record,
        scope_sequential_record,
    )

    inputs = []
    if record["answer"]["breakout"] is None:
        results = scope_sequential_record(record, declarations, **options)
    else:
        results = scope_sequential_breakout_record(
            record, declarations, **options, assembly_inputs=inputs
        )
    return results, inputs


def _day_axis_with_inputs(record, request, *, plan, design, lift_kwargs, configs):
    from tests.evidence_fixtures.temporal import estimate_day_axis_record

    inputs = []
    results = estimate_day_axis_record(
        record,
        request,
        plan=plan,
        design=design,
        lift_kwargs=lift_kwargs,
        configs=configs,
        assembly_inputs=inputs,
    )
    assert len(inputs) == 1
    return results, inputs[0]


def _assert_sequential_snapshot(results, record, declarations, options, inputs):
    from tests.evidence_fixtures.temporal import sequential_record_snapshot
    from tests.evidence_fixtures.transcript import _plain, _restore

    context = declarations.context
    plan = context.plan
    captured = record["answer"]["breakout"]
    if captured is not None:
        assert len(inputs) == 1
        receipt = inputs[0]
        assert receipt.keys() == {"identity_rows", "request", "plan", "source_components"}
        assert receipt["request"] == captured["scope_request"]
        assert receipt["plan"] == _plain(plan)
        assert receipt["source_components"] == captured["source_components"]
        request = _restore(captured["scope_request"])
        _assert_common_snapshot(
            results,
            receipt["identity_rows"],
            request,
            plan,
            route=captured["route"],
            view="breakout",
            source=request["source_name"],
            components=_restore(captured["source_components"]),
        )
        return
    assert inputs == []
    snapshot = sequential_record_snapshot(record)
    roster = _restore(record["answer"]["roster"])
    components = []
    chains = [("assigned", snapshot, roster["integrity_counts"])]
    if snapshot.triggered is not None:
        chains.append(("triggered", snapshot.triggered, snapshot.triggered.assignment_counts))
    for population, chain, counts in chains:
        components.append(
            {
                "kind": "sequential_prefix",
                "metric": None,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": sha256(
                    canonical_json_bytes(
                        {"registration_id": chain.registration_id, "prefix_id": chain.prefix_id}
                    )
                ).hexdigest(),
            }
        )
        if counts is not None:
            components.append(
                {
                    "kind": "assignment_counts",
                    "metric": None,
                    "population": population,
                    "source": None,
                    "dimension": None,
                    "sha256": sha256(canonical_digest_bytes(counts)).hexdigest(),
                }
            )
    source = _composite_input_source(components)
    metrics = [metric.name for metric in options["metrics"]]
    estimands = options["estimands"]
    request = {
        "view": "run",
        "population": "assigned",
        "metrics": None
        if estimands is not None and set(estimands) == {"compliance"}
        else sorted(metrics),
        "estimands": None if estimands is None else sorted(estimands),
        "registration_id": snapshot.registration_id,
        "alpha": plan.alpha,
        "q": plan.q,
        "trigger_declared": context.trigger_name is not None,
        "source_identity": _restore(record["answer"]["source_identity"]),
    }
    method_configs = []
    for config in sorted(options["configs"], key=lambda value: value.metric.name):
        if request["metrics"] is not None and config.metric.name not in metrics:
            continue
        methods = (
            ()
            if config.methods_explicitly_empty
            else (() if config.decision_defaulted else (config.decision_method,))
            + config.sensitivity_methods
        )
        if methods != config.source_methods:
            method_configs.append(
                {
                    "metric": config.metric.name,
                    "decision_method": config.decision_method.model_dump(mode="json"),
                    "sensitivity_methods": [
                        method.model_dump(mode="json") for method in config.sensitivity_methods
                    ],
                }
            )
    if method_configs:
        request["method_configs"] = method_configs
    if context.trigger_name is not None:
        request["trigger_name"] = context.trigger_name
    assert results.source == source
    _assert_snapshot_preimage(
        results,
        {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": "LiftEstimates",
            "source": source,
            "request": request,
        },
    )


def _assert_common_snapshot(
    results, rows, request, plan, *, route, view, source=None, components=()
):
    identity = dict(request)
    if rows and rows[0].get("sequential_result") is not None:
        identity.pop("metrics", None)
        identity.pop("configs", None)
    identity["plan"] = {
        "declared": plan.declared,
        "alpha": plan.alpha,
        "q": plan.q,
        "procedures": {
            name: procedure.model_dump(mode="python") for name, procedure in plan.procedures.items()
        },
    }
    projected_rows = [_input_evidence_projection(row) for row in rows]
    descriptor = {
        "kind": "readout_input_evidence",
        "route": route,
        "source": source,
        "sha256": sha256(
            canonical_json_bytes(sorted(projected_rows, key=canonical_json_bytes))
        ).hexdigest(),
    }
    if components:
        descriptor["components"] = [dict(component) for component in components]
    assert results.source is None
    _assert_snapshot_preimage(
        results,
        {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": type(results).__name__,
            "view": view,
            "request": _snapshot_request(identity),
            "source": descriptor,
        },
    )


def _assert_contrast_snapshot(results, transcript, records, procedures):
    from tests.evidence_fixtures.transcript import _restore

    stats = {
        record["request"]["metric"]["name"]: _restore(record["answer"]["value"])
        for record in records
    }
    source = _composite_input_source(
        [
            {
                "kind": "unit_evidence_rows",
                "metric": metric,
                "population": "assigned",
                "source": None,
                "dimension": None,
                "sha256": sha256(
                    canonical_digest_bytes(_input_evidence_projection(value))
                ).hexdigest(),
            }
            for metric, value in stats.items()
        ]
    )
    fields = (
        "aggregation",
        "probability_ct",
        "randomization_law",
        "independence_grain",
        "washout_steps",
        "carryover_order",
        "observation_steps",
        "retained_steps",
        "identifying_assumption",
        "control_group",
        "treatment_group",
        "n_units",
        "n_cycles",
        "n_blocks",
        "ct_cycles",
        "tc_cycles",
        "minimum_cycles_per_unit",
        "maximum_cycles_per_unit",
    )
    request = {
        "kind": "contrast",
        "metrics": sorted(stats),
        "populations": ["assigned"],
        "procedures": {
            metric: procedures[metric].model_dump(mode="json") for metric in sorted(stats)
        },
        "construction": {
            metric: {field: stats[metric][field] for field in fields} for metric in sorted(stats)
        },
    }
    assert results.source == source
    _assert_snapshot_preimage(
        results,
        {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": "ContrastResults",
            "source_identity": transcript.provenance["source_inputs"][0]["configuration"][
                "source_identity"
            ],
            "source": source,
            "request": request,
        },
    )


def _rebind_source_descriptor(expected, actual, source_id):
    from tests.test_evidence_identity_parity import _compare

    assert actual == _SNAPSHOT_INPUT_SOURCES[source_id]
    assert expected.keys() == actual.keys()
    projected = deepcopy(expected)
    if expected["kind"] == "composite":
        assert (
            expected["sha256"] == sha256(canonical_json_bytes(expected["components"])).hexdigest()
        )
    assert len(expected["sha256"]) == 64 and int(expected["sha256"], 16) >= 0
    projected["sha256"] = actual["sha256"]
    if "components" in expected:
        assert len(expected["components"]) == len(actual["components"])
        for old, current in zip(projected["components"], actual["components"], strict=True):
            assert len(old["sha256"]) == 64 and int(old["sha256"], 16) >= 0
            old["sha256"] = current["sha256"]
    _compare(projected, actual)
    return projected


def _assert_old_assembled_metadata(parts, transcript):
    from increment.estimation.readout_types import ReadoutAssembly, merge_collection_metadata

    assembled = parts[0]
    for part in parts[1:]:
        metadata, source, rows = merge_collection_metadata(assembled, part, scoped_assembly=True)
        assembled = ReadoutAssembly(
            tuple(rows.values()),
            metadata,
            source,
            assembled.sequential_snapshot or part.sequential_snapshot,
        )
    bindings, source_scopes = _assert_old_source_scopes(assembled, transcript)
    _assert_old_metadata_coverage(transcript)
    _assert_old_metadata_payload(
        assembled, transcript, bindings, source_scopes, merged=len(parts) > 1
    )


def _assert_old_source_scopes(results, transcript):
    from tests.test_evidence_identity_parity import (
        _capture,
        _compare,
        _post_capture_absolute_contrast_projection,
    )

    scope = results.metadata.scope
    assert set(scope.by_source) <= _ASSERTED_SNAPSHOT_IDS
    expected_id = (
        next(iter(scope.by_source))
        if len(scope.by_source) == 1
        else "sha256:"
        + sha256(
            canonical_json_bytes(
                {
                    "kind": "increment.readout.concat",
                    "version": 1,
                    "sources": sorted(scope.by_source),
                }
            )
        ).hexdigest()
    )
    assert scope.snapshot_id == expected_id
    old = _capture()["scenarios"][transcript.scenario_id][transcript.ingress]["metadata"]["scope"]
    key = transcript.scenario_id, transcript.ingress

    def coordinate(cells):
        return [
            {name: value for name, value in cell.items() if name != "value_scale"} for cell in cells
        ]

    bindings = {}
    source_scopes = {}
    for source_id, source in scope.by_source.items():
        assert source_id in _ASSERTED_SNAPSHOT_IDS
        actual = source.model_dump(mode="json")
        matching = [
            (old_id, payload)
            for old_id, payload in old["by_source"].items()
            if coordinate(payload["cells"]) == coordinate(actual["cells"])
        ]
        assert len(matching) == 1, (key, coordinate(actual["cells"]))
        old_id, payload = matching[0]
        expected = deepcopy(payload)
        expected["source_snapshot_id"] = source_id
        assert (expected["source"] is None) == (actual["source"] is None)
        if expected["source"] is not None:
            expected["source"] = _rebind_source_descriptor(
                expected["source"], actual["source"], source_id
            )
        actual = _post_capture_absolute_contrast_projection(expected, actual)
        _compare(expected, actual)
        _OLD_METADATA_SCOPES.setdefault(key, set()).add(old_id)
        bindings[old_id] = source_id
        source_scopes[source_id] = expected
    return bindings, source_scopes


def _assert_old_metadata_coverage(transcript):
    from tests.test_evidence_identity_parity import _capture

    key = transcript.scenario_id, transcript.ingress
    expected = _capture()["scenarios"][key[0]][key[1]]["metadata"]["scope"]["by_source"]
    assert _OLD_METADATA_SCOPES[key] == expected.keys()


def _assert_old_metadata_payload(results, transcript, bindings, source_scopes, *, merged):
    from increment.result_envelope import AnalysisEnvelope
    from tests.test_evidence_identity_parity import (
        _capture,
        _compare,
        _metadata_without_modes,
        _post_capture_absolute_contrast_projection,
        _project_post_capture_artifact_digests,
    )

    expected = deepcopy(
        _capture()["scenarios"][transcript.scenario_id][transcript.ingress]["metadata"]
    )
    match AnalysisEnvelope((), results.metadata).to_payload()["metadata"]:
        case dict() as actual:
            pass
        case other:
            raise AssertionError(f"saved metadata is not a mapping: {other!r}")
    scope = expected["scope"]
    scope["snapshot_id"] = results.metadata.scope.snapshot_id
    scope["by_source"] = source_scopes
    for record in expected["cells"]:
        record["source_snapshot_id"] = bindings[record["source_snapshot_id"]]
    _rebind_metadata_families(scope, actual["scope"], bindings)
    artifact_inputs = [
        _METADATA_ARTIFACT_INPUTS[source_id] for source_id in results.metadata.scope.by_source
    ]
    expected_artifacts: dict[str, Any] = (
        artifact_inputs[0]
        if all(value == artifact_inputs[0] for value in artifact_inputs)
        else {"manifest_sha256": None, "relation_digests": []}
    )
    assert results.metadata.artifact_manifest_sha256 == expected_artifacts["manifest_sha256"]
    assert [list(pair) for pair in results.metadata.artifact_relation_digests] == [
        list(pair) for pair in expected_artifacts["relation_digests"]
    ]
    comparable = _post_capture_absolute_contrast_projection(
        expected, _metadata_without_modes(actual)
    )
    comparable = _project_post_capture_artifact_digests(expected, comparable)
    try:
        _compare(_metadata_without_modes(expected), comparable)
    except AssertionError as error:
        error.add_note(f"{transcript.scenario_id}/{transcript.ingress}/assembled metadata")
        raise


def _rebind_metadata_families(expected_scope, actual_scope, bindings):
    def coordinate(family):
        return [
            family.get(name)
            for name in (
                "analysis_population",
                "source_snapshot_id",
                "view",
                "dimension",
                "source",
                "name",
                "identity_metric",
                "identity_look",
                "members",
            )
        ]

    assert len(expected_scope["families"]) == len(actual_scope["families"])
    seen = set()
    for expected in expected_scope["families"]:
        expected["source_snapshot_id"] = bindings[expected["source_snapshot_id"]]
        matching = [
            family
            for family in actual_scope["families"]
            if coordinate(family) == coordinate(expected)
        ]
        assert len(matching) == 1, expected
        actual = matching[0]
        assert actual["family_id"] in _ASSERTED_FAMILY_IDS
        assert actual["family_id"] not in seen
        seen.add(actual["family_id"])
        expected["family_id"] = actual["family_id"]
        if "decision_snapshot_id" in actual:
            assert "decision_snapshot_id" in expected
            assert actual["decision_snapshot_id"] in _ASSERTED_SNAPSHOT_IDS
            expected["decision_snapshot_id"] = actual["decision_snapshot_id"]
    expected_scope["families"].sort(
        key=lambda family: (family["family_id"], family["source_snapshot_id"])
    )


def _assert_decision_snapshot(results, source, request, configs, design, *, source_identity=None):
    from dataclasses import replace

    snapshots = {
        config.metric.name: _snapshot_config(
            replace(
                config,
                sensitivity_methods=(),
                covariates=tuple(config.decision_method.covariates or ()),
            ),
            design,
        )
        for config in sorted(configs, key=lambda item: item.metric.name)
    }
    decision_request = {
        **request,
        "configs": list(snapshots.values()),
        "procedures": {
            name: {
                **procedure,
                "decision_method": snapshots[name]["methods"][0]
                if name in snapshots and snapshots[name]["methods"]
                else procedure["decision_method"],
                "sensitivity_methods": [],
            }
            for name, procedure in request["procedures"].items()
        },
    }
    expected = None
    if source is not None and any(
        config.covariates or any(method.covariates for method in config.source_methods)
        for config in configs
    ):
        preimage = {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": "LiftEstimates",
            "source": source,
            "request": decision_request,
        }
        if source_identity is not None:
            preimage["source_identity"] = source_identity
        digest = "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest()
        if digest != results.metadata.scope.snapshot_id:
            expected = digest
            _ASSERTED_SNAPSHOT_IDS.add(expected)
            _SNAPSHOT_INPUT_SOURCES[expected] = deepcopy(source)
    assert {family.decision_snapshot_id for family in results.metadata.scope.families} <= {expected}


def _randomized_input_source(configs, evidence, roster, population, *, decision=False):
    components = []
    for config in configs:
        value = evidence[config.metric.name]
        parts = value.decision_evidence_components if decision else value.evidence_components
        if not decision:
            parts = parts or ((value.evidence_kind, value.evidence_sha256, value.evidence_count),)
        components.extend(
            {
                "kind": kind,
                "metric": config.metric.name,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": digest,
                "count": count,
            }
            for kind, digest, count in parts
        )
    if roster.counts is not None:
        components.append(
            {
                "kind": "assignment_counts",
                "metric": None,
                "population": population,
                "source": None,
                "dimension": None,
                "sha256": sha256(canonical_digest_bytes(roster.counts)).hexdigest(),
            }
        )
    return _composite_input_source(components)


def _assert_fixed_snapshot(results, transcript, population, configs, design, evidence, roster):
    from increment.decision_wire import compiled_plan_from_dict

    assembly = transcript.provenance["consumer_configuration"]["assembly"]
    plan = compiled_plan_from_dict(assembly["identity_plan"])
    source = _randomized_input_source(configs, evidence, roster, population)
    assert results.source == source
    request = {
        "view": "run",
        "population": population,
        "metrics": [
            config.metric.model_dump(mode="json")
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
        "configs": [
            _snapshot_config(config, design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
        "design": design.model_dump(mode="json"),
        "alpha": plan.alpha,
        "q": plan.q,
        "declared": plan.declared,
        "procedures": {
            name: _snapshot_request(procedure.model_dump(mode="json"), field=f"procedures.{name}")
            for name, procedure in plan.procedures.items()
        },
        "source_identity": assembly["source_identity"],
    }
    trigger = transcript.provenance["consumer_configuration"]["base_context"]["trigger_name"]
    if trigger is not None:
        request.update(trigger_declared=True, trigger_name=trigger)
    _assert_snapshot_preimage(
        results,
        {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": "LiftEstimates",
            "source": source,
            "request": request,
        },
    )
    _assert_decision_snapshot(
        results,
        _randomized_input_source(configs, evidence, roster, population, decision=True),
        request,
        configs,
        design,
    )


def _record_fixed_computation(transcript, population, computation):
    key = transcript.scenario_id, transcript.ingress, population
    _FIXED_COMPUTATIONS.setdefault(key, []).append(computation)


def _method_family_results(
    transcript,
    record,
    config,
    procedure,
    methods,
    design,
    context,
    configuration,
    plan,
    alpha,
):

    import polars as pl

    from increment.estimation.encouragement import ESTIMANDS, _estimate_encouragement
    from increment.estimation.quantile import quantile_lift_from_rows
    from tests.evidence_fixtures.method_replay import (
        ObservationalReplayConfig,
        estimate_observational_transcript,
    )
    from tests.evidence_fixtures.transcript import _restore

    metric = config.metric
    population = record["request"]["population"]
    roles: dict[str, Literal["decision", "sensitivity"]] = {
        method.name: "decision" if index == 0 else "sensitivity"
        for index, method in enumerate(methods)
    }
    options = {
        "metric": metric,
        "methods": methods,
        "design": design,
        "prior": config.prior,
        "alpha": alpha,
        "alternative": procedure.alternative,
        "null_lift": getattr(procedure, "null_lift", 0.0),
        "null_abs": getattr(procedure, "null_abs", None),
        "roles": roles,
        "cluster": context["cluster"],
        "population": population,
        "estimands": configuration["estimands"],
    }
    answer = _restore(record["answer"])
    if design.mechanism == "encouragement":
        rows = answer["rows"] if record["request_type"] == "MetricEvidenceRequest" else answer
        estimands = tuple(
            name for name in (configuration["estimands"] or ESTIMANDS) if name != "compliance"
        )
        computation = _consume_once(
            "encouragement",
            rows,
            options,
            lambda: _estimate_encouragement(
                [metric],
                rows,
                design,
                estimands=estimands,
                methods=list(methods),
                prior=config.prior,
                alpha=alpha,
                alternative=procedure.alternative,
                null_lift=getattr(procedure, "null_lift", 0.0),
                null_abs=getattr(procedure, "null_abs", None),
                family_member=procedure.role == "secondary" and procedure.family.member,
                inference=None,
                cluster=context["cluster"],
                method_roles=roles,
            ),
        )
        _record_fixed_computation(transcript, population, computation)
        from increment.readouts._design_scope import _project_scoped_row

        yield from (
            (_project_scoped_row(row, None, {}), "scalar", population)
            for row in computation.results
        )
        return
    unit_record = next(
        (
            r
            for r in transcript.records
            if r["request_type"] == "UnitRowsRequest"
            and r["request"]["metric"]["name"] == metric.name
            and r["request"]["population"] == population
        ),
        None,
    )
    if unit_record is None:
        assert metric.type == "quantile" and answer["unit_frame"] is not None
        frame = answer["unit_frame"]
    else:
        frame = _restore(unit_record["answer"])["frame"]
    if metric.type == "quantile":
        results = _consume_once(
            "quantile",
            frame,
            options,
            lambda: quantile_lift_from_rows(
                pl.DataFrame(frame["rows"], schema=frame["columns"], orient="row"),
                metric,
                design.control_group,
                design=design,
                authenticate=None,
                methods=list(methods),
                prior=config.prior,
                alpha=alpha,
                inference=None,
                method_roles=roles,
            ),
        )
        from increment.estimation.engine import _lift_decision_bundle

        _record_fixed_computation(
            transcript,
            population,
            _lift_decision_bundle(results, inference=None),
        )
        yield from ((row, "scalar", population) for row in results)
        return
    assert design.mechanism == "observational"
    assert unit_record is not None
    replay_config = ObservationalReplayConfig(
        design=design,
        metrics=(metric,),
        cluster=context["cluster"],
        methods=methods,
        prior=config.prior,
        alpha=alpha,
        alternative=procedure.alternative,
        value_scale=None,
        null_lifts=None,
        null_abs={metric.name: procedure.null_abs}
        if getattr(procedure, "null_abs", None) is not None
        else None,
        alternatives={metric.name: procedure.alternative},
        scales={metric.name: "relative"},
        raise_if_empty=False,
        resolved_method_roles=roles,
        method_roles=roles,
        family_member=procedure.family.member,
        route_tail=None,
        prewarned_unsupported=frozenset(),
    )
    computation = _consume_once(
        "observational",
        {"moments": answer["rows"], "frame": frame},
        replay_config,
        lambda: estimate_observational_transcript(
            transcript,
            (_decode_transcript_request(record),),
            (_decode_transcript_request(unit_record),),
            replay_config,
        ),
    )
    _record_fixed_computation(transcript, population, computation)
    yield from ((row, "scalar", population) for row in computation.results)


def _compliance_results(transcript):
    from increment.estimation.encouragement import estimate_compliance
    from increment.evidence import ComplianceSummary
    from increment.readouts._design_scope import _project_scoped_row
    from tests.evidence_fixtures.transcript import _restore

    configuration, _context, design, _configs, plan = _consumer_context(transcript)
    if design.mechanism != "encouragement":
        return
    for record in transcript.records:
        if record["request_type"] != "ComplianceRequest":
            continue
        payload = _restore(record["answer"])
        summary = ComplianceSummary.from_wire(payload, study_id=payload["study_id"])
        requested = configuration["estimands"] is None or "compliance" in configuration["estimands"]
        options: dict[str, Any] = {
            "design": design,
            "alpha": plan.alpha,
            "inference": None,
            "prior": None,
            "compliance_requested": requested,
        }
        computation = _consume_once(
            "compliance",
            payload,
            options,
            lambda summary=summary, options=options: estimate_compliance(summary, **options),
        )
        _record_fixed_computation(transcript, record["request"]["population"], computation)
        yield from (
            (_project_scoped_row(row, None, {}), "scalar", record["request"]["population"])
            for row in computation.results
        )


def _fixed_coordinate_results(transcript, selected_options=None):
    from increment._analysis_config import effective_methods
    from increment._policy_alpha import resolve_cell_alpha
    from tests.evidence_fixtures.method_replay import METHOD_REPLAYERS
    from tests.evidence_fixtures.transcript import _canonical, _plain, _restore

    configuration, context, design, configs, plan = _consumer_context(transcript)
    by_metric = {config.metric.name: config for config in configs}
    counts_by_population = {
        r["request"]["population"]: r["answer"]["counts"]
        for r in transcript.records
        if r["request_type"] == "CountsRequest" and r["request"]["kind"] == "unit"
    }
    for record in transcript.records:
        if record["request_type"] not in {"MetricEvidenceRequest", "MomentsRequest"}:
            continue
        name = record["request"]["metric"]["name"]
        population = record["request"]["population"]
        if selected_options is not None and (population, name) not in selected_options:
            continue
        n_arms = len(counts_by_population[population]) - 1
        config = by_metric[name]
        procedure = plan.procedures[name]
        if design.mechanism == "observational" and procedure.role == "secondary":
            continue
        alpha = resolve_cell_alpha(plan, procedure, n_arms=n_arms, view=None)
        methods = effective_methods(config, design=design)
        selected = None if selected_options is None else selected_options[(population, name)]
        if selected is not None:
            alpha = selected["alpha"]
            procedure = procedure.model_copy(update={"alternative": selected["alternative"]})
            methods = methods[:1]
        answer = _restore(record["answer"])
        if config.metric.type == "quantile" or design.mechanism in {
            "observational",
            "encouragement",
        }:
            yield from _method_family_results(
                transcript,
                record,
                config,
                procedure,
                methods,
                design,
                context,
                configuration,
                plan,
                alpha,
            )
            continue
        for method in methods:
            role = "decision" if method == config.decision_method else "sensitivity"
            if method.variance_reduction == "cuped_multicovariate":
                fit_records = [
                    r
                    for r in transcript.records
                    if r["request_type"] == "MethodFitRequest"
                    and r["request"]["metric"]["name"] == name
                    and r["request"]["method"] == _plain(method)
                ]
                fitted = [
                    r
                    for r in fit_records
                    if r["answer"]["origin"]
                    in {"warehouse", "retained_fitted", "retained_categorical"}
                ]
                options: dict[str, Any] = {
                    "method_role": role,
                    "alpha": alpha,
                    "alternative": procedure.alternative,
                    "null_lift": getattr(procedure, "null_lift", 0.0),
                    "null_abs": getattr(procedure, "null_abs", None),
                    "prior": config.prior,
                }
                if fitted:
                    assert len(fitted) == 1
                    fit_record = fitted[0]
                    request = _decode_transcript_request(fit_record)
                    kind = (
                        "categorical_cuped"
                        if fit_record["answer"]["origin"] == "retained_categorical"
                        else "fitted_cuped"
                    )
                    computation = _consume_once(
                        kind,
                        fit_record["answer"]["cells"]
                        if kind == "categorical_cuped"
                        else fit_record["answer"]["envelope"],
                        {
                            "metric": request.metric,
                            "method": method,
                            "control": request.control,
                            "arms": request.arm_order,
                            "population": request.population,
                            **options,
                        },
                        lambda kind=kind, request=request, options=options: METHOD_REPLAYERS[kind](
                            transcript, request, **options
                        ),
                    )
                else:
                    assert record["request_type"] == "MetricEvidenceRequest"
                    request = _decode_transcript_request(record)
                    computation = _consume_once(
                        "unit_cuped",
                        {"frame": answer["unit_frame"], "columns": answer["covariate_columns"]},
                        {
                            "metric": request.metric,
                            "method": method,
                            "control": request.control_group,
                            "population": request.population,
                            "covariates": request.covariates,
                            **options,
                        },
                        lambda request=request, method=method, options=options: METHOD_REPLAYERS[
                            "unit_cuped"
                        ](transcript, request, method=method, **options),
                    )
                _record_fixed_computation(transcript, population, computation)
                yield from ((row, "cuped", population) for row in computation.results)
                continue
            rows = (
                dict(answer["method_rows"]).get(method.name, answer["rows"])
                if record["request_type"] == "MetricEvidenceRequest"
                else answer
            )
            arguments = {
                "metric": config.metric,
                "summary": rows,
                "control_group": design.control_group,
                "methods": [method],
                "prior": config.prior,
                "alpha": alpha,
                "alternative": procedure.alternative,
                "null_lift": getattr(procedure, "null_lift", 0.0),
                "null_abs": getattr(procedure, "null_abs", None),
                "preferred_direction": config.metric.declared_preferred_direction,
                "cluster": context["cluster"],
                "method_roles": {method.name: role},
                "summary_population": population,
                "family_member": selected is None
                and procedure.role == "secondary"
                and procedure.family.member,
                "route_tail": None if selected is None else selected["route_tail"],
            }
            computation = _scalar_consumer(_canonical(_plain(arguments)))
            _record_fixed_computation(transcript, population, computation)
            yield from ((row, "scalar", population) for row in computation.results)
    if selected_options is None:
        yield from _compliance_results(transcript)


def _decode_source_context(value: dict[str, Any]):
    from pydantic import TypeAdapter

    from increment.decision_wire import compiled_plan_from_dict
    from increment.evidence import SourceContext
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Metric

    design = {
        "randomized": Randomized,
        "encouragement": Encouragement,
        "observational": Observational,
    }[value["design"]["mechanism"]].model_validate(value["design"])
    arguments: dict[str, Any] = {
        **value,
        "design": design,
        "plan": compiled_plan_from_dict(value["plan"]),
        "metrics": tuple(TypeAdapter(Metric).validate_python(m) for m in value["metrics"]),
        "configs": tuple(_decode_config(config) for config in value["configs"]),
    }
    return SourceContext(**arguments)


def _metric_evidence_inputs(transcript, population, design):
    import pandas as pd

    from increment._evidence_types import MetricEvidence
    from increment.evidence import MethodFitAnswer
    from tests.evidence_fixtures.transcript import _restore

    evidence = {}
    for record in transcript.records:
        if record["request_type"] not in {"MetricEvidenceRequest", "MomentsRequest"}:
            continue
        if record["request"]["population"] != population:
            continue
        name = record["request"]["metric"]["name"]
        if record["request_type"] == "MetricEvidenceRequest":
            answer = _restore(record["answer"])
            frame = answer["unit_frame"]
            if frame is not None:
                answer["unit_frame"] = pd.DataFrame(frame["rows"], columns=frame["columns"])
            answer["method_fit_answers"] = tuple(
                (
                    _decode_transcript_request(
                        {"request_type": "MethodFitRequest", "request": request}
                    ),
                    MethodFitAnswer(**fit),
                )
                for request, fit in answer["method_fit_answers"]
            )
            evidence[name] = MetricEvidence(**answer)
        else:
            rows = tuple(_restore(record["answer"]))
            arms = frozenset(str(row["group_id"]) for row in rows)
            digest, count = _digest_input_rows(rows)
            evidence[name] = MetricEvidence(
                rows=rows,
                unit_frame=None,
                observed_arms=arms,
                n_treatment_arms=len(arms - {design.control_group}),
                evidence_kind="moments_rows",
                evidence_sha256=digest,
                evidence_count=count,
                decision_evidence_components=(("moments_rows", digest, count),),
                evidence_components=(("moments_rows", digest, count),),
                cluster=transcript.provenance["consumer_configuration"]["context"]["cluster"],
                control_group=design.control_group,
            )
    return evidence


def _project_randomized_family(transcript, items, population, configs, plan, design, evidence):
    from increment._analysis_config import effective_methods
    from increment.estimation.conversion_route import fcr_route_tail
    from increment.estimation.engine import merge_decision_computations
    from increment.estimation.family import decision_cells, select_family
    from increment.estimation.results import _fcr_alpha_for, open_bound_from_two_sided_at_target
    from increment.readouts._common import _joint_relative_rows
    from increment.readouts._randomized import (
        _partition_secondary_family,
        _secondary_family_cells,
        _secondary_rows_without_fcr,
        _stamp_fcr_selected_rows,
    )

    entries = [
        (
            config.metric,
            config,
            plan.procedures[config.metric.name],
            config.metric.type == "quantile",
            list(effective_methods(config, design=design)),
        )
        for config in configs
        if plan.procedures[config.metric.name].role == "secondary"
    ]
    family, _non_family = _partition_secondary_family(entries)
    if not family:
        return items
    names = {metric.name for metric, *_rest in family}
    nominal = {name: [row for row, _consumer in items if row.metric == name] for name in names}
    groups = {name: set(evidence[name].observed_arms) - {design.control_group} for name in names}
    computations = _FIXED_COMPUTATIONS[(transcript.scenario_id, transcript.ingress, population)]
    outcome = select_family(
        decision_cells(_secondary_family_cells(family, nominal, groups)),
        plan.q,
        None,
        plan.alpha,
        computation=merge_decision_computations(computations),
    )
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_procedure": "bh",
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
        "family_size": outcome.n_family,
        "family_nominal_alpha": plan.alpha,
    }
    if outcome.fcr_alpha is None:
        projected = _secondary_rows_without_fcr(family, nominal, outcome, family_record)
    else:
        selected_names = {key.metric for key in outcome.selected}
        overrides = {}
        for metric, _config, procedure, is_quantile, _methods in family:
            if metric.name not in selected_names:
                continue
            alternative = (
                "two-sided" if _joint_relative_rows(nominal[metric.name]) else procedure.alternative
            )
            overrides[(population, metric.name)] = {
                "alpha": _fcr_alpha_for(alternative, outcome.fcr_alpha),
                "alternative": alternative,
                "route_tail": None
                if is_quantile
                else fcr_route_tail(
                    procedure.alternative,
                    outcome.fcr_alpha,
                ),
            }
        reestimated = {}
        for row, _consumer, _population in _fixed_coordinate_results(transcript, overrides):
            reestimated.setdefault(row.metric, []).append(open_bound_from_two_sided_at_target(row))
        projected = _stamp_fcr_selected_rows(
            family,
            nominal,
            reestimated,
            outcome.selected,
            outcome,
            family_record,
        )
    projected_by_key = {_result_coordinate(row.model_dump(mode="json")): row for row in projected}
    return [
        (projected_by_key[_result_coordinate(row.model_dump(mode="json"))], consumer)
        if row.metric in names
        else (row, consumer)
        for row, consumer in items
    ]


def _project_design_results(
    transcript, declaration, items, population, configs, plan, design, evidence
):
    import pandas as pd

    from increment.evidence import ComplianceSummary
    from increment.readouts._design_scope import (
        ResolvedRoster,
        encouragement_expected,
        observational_expected,
        scope_design_results,
    )
    from increment.readouts._run import _observational_adjustment_identity
    from tests.evidence_fixtures.transcript import _restore

    configuration = transcript.provenance["consumer_configuration"]
    selected = tuple(config.metric for config in configs)
    rows = [row for row, _consumer in items]
    observed = {name: set(value.observed_arms) for name, value in evidence.items()}
    summary = None
    extra = {
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(selected, key=lambda m: m.name)
        ],
        "configs": [
            _snapshot_config(config, design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
    }
    if design.mechanism == "encouragement":
        matching = [
            record
            for record in transcript.records
            if record["request_type"] == "ComplianceRequest"
            and record["request"]["population"] == population
        ]
        if matching:
            assert len(matching) == 1
            payload = _restore(matching[0]["answer"])
            summary = ComplianceSummary.from_wire(payload, study_id=payload["study_id"])
            observed["uptake"] = {arm.group_id for arm in summary.arms}
        from increment.estimation.engine import merge_decision_computations
        from increment.readouts._encouragement import _select_encouragement_family

        computations = _FIXED_COMPUTATIONS[(transcript.scenario_id, transcript.ingress, population)]
        rows = _select_encouragement_family(
            rows,
            computation=merge_decision_computations(computations),
            metrics=selected,
            configs=configs,
            plan=plan,
            design=design,
            rows_by_metric={name: value.rows for name, value in evidence.items()},
            evidence_by_metric=evidence,
            estimands=configuration["estimands"],
            cluster=declaration.context.cluster,
            computations=computations,
        )

        def expected_for(arms):
            return encouragement_expected(
                selected,
                configs,
                design,
                plan,
                estimands=configuration["estimands"],
                arms=arms,
                cluster=declaration.context.cluster,
            )
    else:

        def unit_frame(metric, _covariates):
            record = next(
                record
                for record in transcript.records
                if record["request_type"] == "UnitRowsRequest"
                and record["request"]["metric"]["name"] == metric.name
                and record["request"]["population"] == population
            )
            frame = _restore(record["answer"])["frame"]
            return pd.DataFrame(frame["rows"], columns=frame["columns"])

        from increment._analysis_config import effective_methods
        from increment.estimation.adjust import ObservationalEvidence
        from increment.readouts._observational import _estimate_observational_secondary_family

        secondary = tuple(
            metric for metric in selected if plan.procedures[metric.name].role == "secondary"
        )
        if secondary:

            def estimate_family():
                computations = []
                family_rows = _estimate_observational_secondary_family(
                    declaration,
                    secondary,
                    {config.metric.name: config for config in configs},
                    design,
                    plan,
                    methods_by_metric={
                        config.metric.name: list(effective_methods(config, design=design))
                        for config in configs
                    },
                    value_scale=None,
                    resolved_null_abs={
                        metric.name: plan.procedures[metric.name].null_abs
                        for metric in selected
                        if getattr(plan.procedures[metric.name], "null_abs", None) is not None
                    },
                    resolved_alternatives={
                        metric.name: plan.procedures[metric.name].alternative for metric in selected
                    },
                    evidence=ObservationalEvidence(
                        {name: value.rows for name, value in evidence.items()},
                        declaration.context.cluster,
                    ),
                    unit_frame_memo=unit_frame,
                    computations=computations,
                )
                return family_rows, computations

            family_rows, computations = _consume_once(
                "observational_family",
                {
                    "moments": {name: value.rows for name, value in evidence.items()},
                    "units": [
                        record
                        for record in transcript.records
                        if record["request_type"] == "UnitRowsRequest"
                        and record["request"]["population"] == population
                    ],
                },
                {"configs": configs, "plan": plan, "design": design, "population": population},
                estimate_family,
            )
            rows.extend(family_rows)
            _FIXED_COMPUTATIONS.setdefault(
                (transcript.scenario_id, transcript.ingress, population),
                [],
            ).extend(computations)
        expected_adjustment = _observational_input_identity(
            transcript,
            selected,
            configs,
            design,
            rows,
            population,
        )
        actual_adjustment = _observational_adjustment_identity(
            declaration,
            selected,
            configs,
            design,
            rows,
            unit_frame,
        )
        assert actual_adjustment == expected_adjustment
        extra.update(
            value_scale=None,
            adjustment_input_identity=actual_adjustment,
        )

        def expected_for(arms):
            return observational_expected(
                selected, configs, design, plan, value_scale=None, arms=arms
            )

    roster = configuration["assembly"]["rosters"][population]
    assert design.mechanism == "observational" or roster is not None
    result = scope_design_results(
        declaration,
        design,
        plan,
        rows,
        _FIXED_COMPUTATIONS[(transcript.scenario_id, transcript.ingress, population)],
        configs=configs,
        population=population,
        expected_for=expected_for,
        evidence_rows={name: value.rows for name, value in evidence.items()},
        observed_by_metric=observed,
        compliance_summary=summary,
        estimands=configuration["estimands"],
        use_counts=design.mechanism != "observational",
        base_roster=None if roster is None else ResolvedRoster(**roster),
        request_extra=extra,
        method_evidence=evidence if design.mechanism == "encouragement" else None,
    )
    request = {
        "view": "run",
        "population": population,
        "mechanism": design.mechanism,
        "design": design.model_dump(mode="json"),
        "estimands": None
        if configuration["estimands"] is None
        else sorted(configuration["estimands"]),
        "alpha": plan.alpha,
        "q": plan.q,
        "declared": plan.declared,
        "procedures": {
            name: _snapshot_request(procedure.model_dump(mode="python"), field=f"procedures.{name}")
            for name, procedure in plan.procedures.items()
        },
        **extra,
    }
    if declaration.context.trigger_name is not None:
        request.update(trigger_declared=True, trigger_name=declaration.context.trigger_name)
    expected_source = _design_input_source(
        transcript, population, design, evidence, summary, roster
    )
    assert result.source == expected_source
    _assert_snapshot_preimage(
        result,
        {
            "kind": "increment.readout.snapshot",
            "version": 2,
            "collection": "LiftEstimates",
            "source_identity": configuration["assembly"]["source_identity"],
            "source": expected_source,
            "request": request,
        },
    )
    _assert_decision_snapshot(
        result,
        _design_input_source(
            transcript, population, design, evidence, summary, roster, decision=True
        )
        if design.mechanism == "encouragement"
        else None,
        request,
        configs,
        design,
        source_identity=configuration["assembly"]["source_identity"],
    )
    return result


def _project_fixed_results(transcript):
    from increment._evidence_types import ArtifactProvenance
    from increment.decision_wire import compiled_plan_from_dict
    from increment.readouts._design_scope import ResolvedRoster
    from increment.readouts._randomized_scope import _scoped_randomized_results
    from tests.evidence_fixtures.method_replay import ReadoutDeclaration

    configuration, _context, design, configs, plan = _consumer_context(transcript)
    for key in list(_FIXED_COMPUTATIONS):
        if key[:2] == (transcript.scenario_id, transcript.ingress):
            del _FIXED_COMPUTATIONS[key]
    consumed = list(_fixed_coordinate_results(transcript))
    assembly = configuration["assembly"]
    declaration = ReadoutDeclaration(
        _decode_source_context(configuration["base_context"]),
        assembly["source_identity"],
    )
    populations = tuple(dict.fromkeys(scope["population"] for scope in configuration["scopes"]))
    parts = []
    for population in populations:
        roster = assembly["rosters"][population]
        evidence = _metric_evidence_inputs(transcript, population, design)
        selected = tuple(config.metric for config in configs)
        items = [(row, consumer) for row, consumer, member in consumed if member == population]
        calculation_plan = compiled_plan_from_dict(assembly["calculation_plans"][population])
        if design.mechanism == "randomized":
            assert roster is not None, "captured randomized roster is required before assembly"
            base_roster = ResolvedRoster(**roster)
            items = _project_randomized_family(
                transcript,
                items,
                population,
                configs,
                calculation_plan,
                design,
                evidence,
            )
            results = _scoped_randomized_results(
                declaration,
                selected,
                configs,
                design,
                calculation_plan,
                [row for row, _consumer in items],
                _FIXED_COMPUTATIONS[(transcript.scenario_id, transcript.ingress, population)],
                {name: set(value.observed_arms) for name, value in evidence.items()},
                evidence,
                ArtifactProvenance(**assembly["artifact_provenance"]),
                population,
                base_roster,
                identity_source=declaration,
                identity_plan=compiled_plan_from_dict(assembly["identity_plan"]),
            )
            _assert_fixed_snapshot(
                results, transcript, population, configs, design, evidence, base_roster
            )
        else:
            results = _project_design_results(
                transcript,
                declaration,
                items,
                population,
                configs,
                calculation_plan,
                design,
                evidence,
            )
        parts.append(results)
        _register_readout_identities(
            results,
            artifact_provenance=assembly["artifact_provenance"]
            if design.mechanism == "randomized"
            else None,
        )
        consumers = {
            _result_coordinate(row.model_dump(mode="json")): consumer for row, consumer in items
        }
        if design.mechanism == "observational":
            consumers.update(
                (_result_coordinate(row.model_dump(mode="json")), "scalar") for row in results
            )
        for row in results:
            cells = row.as_cells(results.metadata)
            assert len(cells) == 1
            _FIXED_PUBLISHED_ROWS[
                transcript.scenario_id,
                transcript.ingress,
                population,
                _result_coordinate(row.model_dump(mode="json")),
            ] = cells[0].to_payload()["retained_row"]
            yield row, consumers[_result_coordinate(row.model_dump(mode="json"))], population
    _assert_old_assembled_metadata(parts, transcript)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid(
    "All scalar and CUPED frozen statistical representations and complete configurations"
)
def test_frozen_scalar_and_cuped_consumers_match_all_old_owned_fields():
    from tests.evidence_fixtures.replay import assert_estimator_matches_old_row

    for entry in scenario_roster():
        if entry.disposition != "source" or entry.family not in {
            "mean",
            "cuped",
            "cluster",
            "conversion",
            "ratio",
            "quantile",
            "observational",
            "encouragement",
        }:
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        if any(
            r["request_type"]
            in {
                "DayAxisRequest",
                "SequentialCaptureRequest",
                "ContrastStatsRequest",
            }
            for r in transcript.records
        ) or any(
            scope["view"] == "breakout"
            for scope in transcript.provenance["consumer_configuration"]["scopes"]
        ):
            continue
        expected = _old_coordinate_rows(entry.scenario_id, entry.ingress)
        expected_by_key = {
            (_result_coordinate(row), row["analysis_population"]): row for row in expected
        }
        assert len(expected_by_key) == len(expected)
        observed = set()
        for result, consumer, population in _project_fixed_results(transcript):
            key = _result_coordinate(result.model_dump(mode="json")), population
            assert key in expected_by_key, (entry.scenario_id, entry.ingress, key)
            assert key not in observed, (entry.scenario_id, entry.ingress, key)
            try:
                assert_estimator_matches_old_row(result, expected_by_key[key], consumer=consumer)
                published = _FIXED_PUBLISHED_ROWS[
                    entry.scenario_id,
                    entry.ingress,
                    population,
                    key[0],
                ]
                assert published is not None
                _assert_old_readout_rows(
                    (published,),
                    (expected_by_key[key],),
                    nominal_alpha=_consumer_context(transcript)[4].alpha,
                )
            except AssertionError as error:
                error.add_note(f"{entry.scenario_id}/{entry.ingress}/{key!r}")
                raise
            observed.add(key)
        assert observed == expected_by_key.keys(), (entry.scenario_id, entry.ingress)
        _assert_old_metadata_coverage(transcript)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("Complete captured sequential and switchback statistical representations")
def test_frozen_sequential_and_contrast_consumers_match_old_rows():
    from fractions import Fraction

    from increment.estimation.decision_types import ContrastDecisionProcedure
    from tests.evidence_fixtures.method_replay import ReadoutDeclaration
    from tests.evidence_fixtures.temporal import (
        assert_switchback_unit_t_reference,
        scope_contrast_records,
    )

    for entry in scenario_roster():
        if entry.disposition != "source":
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        records = [
            record
            for record in transcript.records
            if record["request_type"] in {"SequentialCaptureRequest", "ContrastStatsRequest"}
        ]
        if not records:
            continue
        configuration = transcript.provenance["consumer_configuration"]
        if records[0]["request_type"] == "ContrastStatsRequest":
            procedures = {
                name: ContrastDecisionProcedure.model_validate(procedure)
                for name, procedure in configuration["context"]["procedures"].items()
            }
            source_identity = transcript.provenance["source_inputs"][0]["configuration"][
                "source_identity"
            ]
            results = _consume_once(
                "contrast_readout",
                records,
                {"procedures": procedures, "source_identity": source_identity},
                lambda records=records, procedures=procedures, source_identity=source_identity, transcript=transcript: (
                    scope_contrast_records(
                        records,
                        procedures,
                        source_identity_record=source_identity,
                        **transcript.provenance["coherence"],
                    )
                ),
            )
            assert entry.scenario_id == "audit-switchback-neighboring-integers"
            assert len(records) == len(results) == 1
            assert_switchback_unit_t_reference(
                records[0],
                results[0],
                unit_deltas={
                    "a": Fraction(2**53 + 1) - Fraction(2**53),
                    "b": Fraction(2**53 + 2) - Fraction(2**53),
                },
            )
        else:
            _configuration, _context, _design, configs, _plan = _consumer_context(transcript)
            assert len(records) == 1
            record = records[0]
            declarations = ReadoutDeclaration(
                _decode_source_context(configuration["base_context"]),
                record["answer"]["source_identity"],
            )
            options = {
                "metrics": tuple(config.metric for config in configs),
                "estimands": configuration["estimands"],
                "configs": configs,
            }
            results, assembly_inputs = _consume_once(
                "sequential_readout",
                record["answer"],
                {"base_context": configuration["base_context"], **options},
                lambda record=record, declarations=declarations, options=options: (
                    _sequential_scope_with_inputs(record, declarations, options)
                ),
            )
        if records[0]["request_type"] == "ContrastStatsRequest":
            _assert_contrast_snapshot(results, transcript, records, procedures)
        else:
            _assert_sequential_snapshot(results, record, declarations, options, assembly_inputs)
        _register_readout_identities(
            results,
            artifact_provenance=records[0]["answer"]["breakout"]["artifact_provenance"]
            if records[0]["request_type"] == "SequentialCaptureRequest"
            and records[0]["answer"]["breakout"] is not None
            else None,
        )
        _assert_old_assembled_metadata([results], transcript)
        if records[0]["request_type"] == "ContrastStatsRequest" or (
            records[0]["request_type"] == "SequentialCaptureRequest"
            and records[0]["answer"]["breakout"] is not None
        ):
            from tests.test_evidence_identity_parity import _capture

            expected = _capture()["scenarios"][entry.scenario_id][entry.ingress]["daily_values"]
        else:
            expected = _old_coordinate_rows(entry.scenario_id, entry.ingress)
        expected_by_key = {_full_readout_coordinate(row): row for row in expected}
        assert len(expected_by_key) == len(expected) == len(results), (
            entry.scenario_id,
            entry.ingress,
        )
        observed = set()
        for result in results:
            key = _full_readout_coordinate(result.model_dump(mode="json"))
            assert key in expected_by_key, (entry.scenario_id, entry.ingress, key)
            assert key not in observed
            _assert_old_readout_rows((result,), (expected_by_key[key],))
            observed.add(key)
        assert observed == expected_by_key.keys()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("Every raw-winsor input representation and complete inference configuration")
def test_frozen_raw_winsor_consumers_preserve_unavailable_sampling_and_raw_reference():
    import math
    from fractions import Fraction

    from scipy.special import ndtri_exp

    from increment._policy_alpha import resolve_cell_alpha
    from increment.winsor import BootstrapReference, InfluenceReference, WinsorRawState
    from tests.estimation.test_winsor_zero_inclusive import _brute_influence, _brute_point
    from tests.evidence_fixtures.method_replay import WinsorReplayConfig, estimate_winsor_transcript
    from tests.evidence_fixtures.transcript import _restore
    from tests.test_evidence_identity_parity import _capture

    for entry in scenario_roster():
        if entry.disposition != "source" or entry.family != "winsor":
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        _configuration, context, design, configs, plan = _consumer_context(transcript)
        old = _capture()["scenarios"][entry.scenario_id][entry.ingress]
        old_cells = [cell["result_cell"] for cell in old["cells"].values()]
        computed = []
        for record in transcript.records:
            if record["request_type"] != "MetricEvidenceRequest":
                continue
            answer = _restore(record["answer"])
            raw = WinsorRawState.model_validate(answer["rows"][0]["_winsor_raw_state"])
            references = answer["rows"][0]["_winsor_references"]
            config = next(
                c for c in configs if c.metric.name == record["request"]["metric"]["name"]
            )
            metric = config.metric
            request = next(
                _decode_transcript_request(r)
                for r in transcript.records
                if r["request_type"] == "UnitRowsRequest"
                and r["request"]["metric"]["name"] == metric.name
            )
            rows = transcript.answer_for(request)["frame"]["rows"]
            arms = {arm.group_id: arm.values for arm in raw.arms}
            assert arms == {
                group: tuple(sorted(row["y"] for row in rows if row["group_id"] == group))
                for group in arms
            }
            procedure = plan.procedures[metric.name]
            alpha = resolve_cell_alpha(plan, procedure, n_arms=len(arms) - 1, view=None)
            for treatment in sorted(set(arms) - {design.control_group}):
                reference_class = (
                    InfluenceReference
                    if raw.executed_method == "influence-normal-v1"
                    else BootstrapReference
                )
                reference = reference_class.model_validate(references[(metric.name, treatment)])
                options = WinsorReplayConfig(
                    study_id=context["study_id"],
                    missingness=raw.missingness,
                    population=request.population,
                    control_group=design.control_group,
                    treatment_group=treatment,
                    inference=raw.inference,
                    support=raw.support,
                    alpha=alpha,
                    method=config.decision_method.name,
                    method_role="decision",
                    null_lift=getattr(procedure, "null_lift", 0.0),
                    null_abs=getattr(procedure, "null_abs", None),
                    preferred_direction=metric.declared_preferred_direction,
                    reference=reference,
                )
                result = _consume_once(
                    "raw_winsor",
                    {"frame": transcript.answer_for(request)["frame"], "metric": metric},
                    options,
                    lambda transcript=transcript, request=request, metric=metric, options=options: (
                        estimate_winsor_transcript(transcript, request, metric, options)
                    ),
                )
                computed.append(result)
                cutoff, relative, additive = _brute_point(
                    arms,
                    raw.quantile,
                    design.control_group,
                    treatment,
                )
                _density, se_log, se_add = _brute_influence(
                    arms,
                    raw.quantile,
                    design.control_group,
                    treatment,
                    cutoff,
                )
                assert result.winsor_upper_bound == pytest.approx(cutoff, rel=1e-12)
                assert result.lift.value == pytest.approx(relative, rel=1e-12, abs=1e-14)
                assert result.abs_diff == pytest.approx(additive, rel=1e-12, abs=1e-14)
                assert reference.log_relative.se == pytest.approx(se_log, rel=1e-12)
                assert reference.additive.se == pytest.approx(se_add, rel=1e-12)
                for series, interval, is_relative in (
                    (reference.log_relative, result.confidence_set.relative, True),
                    (reference.additive, result.confidence_set.additive, False),
                ):
                    if isinstance(reference, BootstrapReference):
                        assert reference.failure_indices == ()
                        values = (
                            reference.log_relative.roots
                            if is_relative
                            else reference.additive.roots
                        )
                        assert all(value is not None for value in values)
                        roots = sorted(cast(tuple[float, ...], values))
                        rank = math.floor(Fraction(alpha) * (reference.spec.replicates + 1) / 2)
                        lower = series.point - roots[-rank] * series.se
                        upper = series.point - roots[rank - 1] * series.se
                    else:
                        z = -float(ndtri_exp(math.log(alpha) - math.log(2)))
                        lower, upper = series.point - z * series.se, series.point + z * series.se
                    if is_relative:
                        lower, upper = math.expm1(lower), math.expm1(upper)
                    assert interval.lower.value == pytest.approx(lower, rel=1e-12, abs=1e-14)
                    assert interval.upper.value == pytest.approx(upper, rel=1e-12, abs=1e-14)
        _assert_winsor_unavailable(computed, old_cells)


def _assert_winsor_unavailable(computed, old_cells):
    from increment.estimation.engine import _lift_decision_bundle

    computation = _lift_decision_bundle(computed, inference=None)
    assert not computation.evidence
    assert len(computation.failures) == len(computed) == len(old_cells)
    assert all(cell["retained_row"] is None and cell["summary"] is None for cell in old_cells)
    for failure in computation.failures.values():
        matches = [
            cell["components"]["sampling"]["status"]
            for cell in old_cells
            if cell["components"]["sampling"]["status"]["context"] == dict(failure.context)
        ]
        assert matches and all(status == matches[0] for status in matches)
        status = matches[0]
        assert status["state"] == "unavailable" and status["code"] == failure.code
        assert status["advisories"] == []
        assert (
            status["reason"] == "percentile-winsorized confidence set (a single-level construction)"
        )


def _breakout_source_partitions(record):
    declared = record["request"]["breakouts"]
    observed = record["answer"]["slices"]
    if record["answer"]["route"] == "unsupported":
        assert observed == []
        return tuple((identity, identity["source"]) for identity in declared)
    assert {json.dumps(item["identity"], sort_keys=True) for item in observed} == {
        json.dumps(identity, sort_keys=True) for identity in declared
    }
    return tuple((item["identity"], item["stamp_source"]) for item in observed)


def _full_readout_coordinate(row):
    if row.get("assignment") == "switchback":
        return tuple(
            row.get(field)
            for field in (
                "metric",
                "control_group",
                "treatment_group",
                "method",
                "estimand",
                "analysis_population",
            )
        )
    return tuple(
        row.get(field)
        for field in (
            "metric",
            "group_id",
            "method",
            "estimand",
            "value_scale",
            "dimension",
            "dimension_value",
            "source",
            "analysis_population",
            "ds",
        )
    )


def _assert_old_readout_rows(actual, expected, *, nominal_alpha=None):
    from tests.evidence_fixtures.replay import _assert_projected_value
    from tests.parity_harness.expected_differences import without_binomial_arm_facts
    from tests.test_evidence_identity_parity import _retained_capture_projection

    actual = [row if isinstance(row, dict) else row.model_dump(mode="json") for row in actual]
    assert len(actual) == len(expected)
    assert [_full_readout_coordinate(row) for row in actual] == [
        _full_readout_coordinate(row) for row in expected
    ]
    for left, right in zip(actual, expected, strict=True):
        if (
            right.get("family_nominal_alpha") is None
            and left.get("family_nominal_alpha") is not None
        ):
            assert left["discovery"] is True and left["family_procedure"] == "bh"
            assert nominal_alpha is not None
            _assert_projected_value(
                "family_nominal_alpha", left["family_nominal_alpha"], nominal_alpha
            )
        if left.get("allocation_alpha") is not None:
            assert left["discovery"] is True
            assert nominal_alpha is not None
            maximum = nominal_alpha * (2.0 if left["alternative"] in {"less", "greater"} else 1.0)
            assert 0.0 < left["allocation_alpha"] <= maximum
        left = without_binomial_arm_facts(right, left)
        left, _compatibility_fields = _retained_capture_projection(right, left)
        extra = left.keys() - right.keys()
        if extra:
            assert left.get("assignment") == "switchback" and extra == {"value_scale"}
            assert left["value_scale"] == "absolute"
        assert right.keys() <= left.keys(), right.keys() - left.keys()
        for field, value in right.items():
            if field in {"source_snapshot_id", "family_id"}:
                source_id = left["source_snapshot_id"]
                identity = source_id, _full_readout_coordinate(left)
                if source_id is None:
                    assert value is None and left[field] is None
                else:
                    assert identity in _OWNED_READOUT_IDENTITIES
                    assert left.get("family_id") == _OWNED_READOUT_IDENTITIES[identity]
                    assert (left[field] is None) == (value is None)
            else:
                _assert_projected_value(field, left[field], value)


def _breakout_scope_data(transcript, identity, source):
    matching = [
        (receipt, components)
        for receipt, components in _audit_breakout_scope_inputs(transcript)
        if receipt["identity"] == identity and receipt["stamp_source"] == source
    ]
    assert len(matching) == 1, (transcript.scenario_id, transcript.ingress, identity, source)
    return matching[0]


def _breakout_readout_inputs(receipt):
    import pandas as pd

    from increment.evidence import MethodFitAnswer
    from tests.evidence_fixtures.replay import _coded_error, _fitted_cuped_envelope
    from tests.evidence_fixtures.transcript import _restore

    fits = {}
    frames = {}
    for entry in receipt["entries"]:
        record = entry["record"]
        request = _decode_transcript_request(record)
        if record["request_type"] == "MethodFitRequest":
            answer = record["answer"]
            failure = None if answer["failure"] is None else _coded_error(answer["failure"])
            envelope = (
                _fitted_cuped_envelope(answer, request)
                if answer["envelope"] is not None and failure is None
                else None
            )
            assert answer["cells"] is None
            typed = MethodFitAnswer(
                origin=answer["origin"],
                envelope=envelope,
                cells=None,
                failure=failure,
                fit_identity=tuple(_restore(answer["fit_identity"])),
                evidence_sha256=answer["evidence_sha256"],
            )
            fits.setdefault(request.metric.name, []).append((request, typed))
        else:
            assert entry["usage"] == "local_breakout_unit_frame"
            answer = _restore(record["answer"])
            dimension = receipt["identity"]["property"]
            names = request.covariates
            alias = "__breakout_dimension"
            while alias in names:
                alias += "_"
            if answer["unit_frame_has_dimension"]:
                requested = tuple(dict.fromkeys((dimension, *names)))
                columns = {}
                for name in requested:
                    column = name
                    if name in {"unit_id", "group_id", "y", "y_den"}:
                        column = f"__covariate_{name}"
                        while column in requested or column in columns.values():
                            column += "_"
                    columns[name] = column
                column = columns[dimension]
            else:
                column = alias
            segmented = {}
            for row in answer["frame"]["rows"]:
                value = row[column]
                segment = (
                    "__null__"
                    if value is None
                    else str(value).lower()
                    if isinstance(value, bool)
                    else str(value)
                )
                values = dict(row)
                if column == alias:
                    values.pop(alias)
                segmented.setdefault(segment, []).append(values)
            frames[request.metric.name] = {
                segment: pd.DataFrame(rows) for segment, rows in segmented.items()
            }
    return {name: tuple(answers) for name, answers in fits.items()}, frames


def _scope_breakout_inputs(transcript, rows, identity, source):
    from increment._evidence_types import ArtifactProvenance
    from increment.breakout.estimates import BreakoutEstimates
    from increment.readouts._multiplicity_scope import scoped_collection
    from tests.evidence_fixtures.transcript import _restore

    receipt, components = _breakout_scope_data(transcript, identity, source)
    context = _decode_source_context(receipt["context"])
    configs, plan, design = context.configs, context.plan, context.design
    request = _restore(receipt["request"])
    result = scoped_collection(
        rows,
        BreakoutEstimates,
        plan,
        configs,
        request,
        metrics=tuple(config.metric for config in configs),
        route=receipt["route"],
        view="breakout",
        design=design,
        dimension=request["dimension"],
        source=request["source_name"],
        source_components=components,
        artifact_provenance=ArtifactProvenance(**_restore(receipt["artifact_provenance"])),
    )
    _assert_common_snapshot(
        result,
        [row.model_dump(mode="json", exclude={"source_snapshot_id"}) for row in rows],
        request,
        plan,
        route=receipt["route"],
        view="breakout",
        source=request["source_name"],
        components=components,
    )
    _register_readout_identities(result, artifact_provenance=receipt["artifact_provenance"])
    if _old_daily_consumer(transcript.scenario_id, transcript.ingress) is None:
        assert components
        _assert_current_input_scope_metadata(result, rows, receipt)
    return result


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("All captured source-qualified breakout inputs and full compiled options")
def test_frozen_breakout_consumers_match_complete_ordered_old_rows():
    from increment._analysis_config import effective_methods
    from increment._evidence_types import MetricEvidence
    from increment.breakout.estimates import run_breakout
    from tests.evidence_fixtures.transcript import _restore
    from tests.test_evidence_identity_parity import _capture

    for entry in scenario_roster():
        if entry.disposition != "source" or entry.temporal:
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        slice_records = [
            record
            for record in transcript.records
            if record["request_type"] == "BreakoutSlicesRequest"
        ]
        if _old_daily_consumer(entry.scenario_id, entry.ingress) == "breakout":
            assert slice_records
        if not slice_records:
            continue
        configuration, context, design, configs, plan = _consumer_context(transcript)
        if design.mechanism == "encouragement":
            continue
        policy = plan.view_policies.randomized_breakout
        methods_by_metric = {
            config.metric.name: list(effective_methods(config, design=design)) for config in configs
        }
        roles = {
            config.metric.name: {
                method.name: "decision" if method == config.decision_method else "sensitivity"
                for method in methods_by_metric[config.metric.name]
            }
            for config in configs
        }
        results = []
        parts = []
        for slice_record in slice_records:
            for identity, stamp_source in _breakout_source_partitions(slice_record):
                receipt, _components = _breakout_scope_data(transcript, identity, stamp_source)
                scoped_context = _decode_source_context(receipt["context"])
                configs, plan, design = (
                    scoped_context.configs,
                    scoped_context.plan,
                    scoped_context.design,
                )
                policy = plan.view_policies.randomized_breakout
                methods_by_metric = {
                    config.metric.name: list(effective_methods(config, design=design))
                    for config in configs
                }
                roles = {
                    config.metric.name: {
                        method.name: "decision"
                        if method == config.decision_method
                        else "sensitivity"
                        for method in methods_by_metric[config.metric.name]
                    }
                    for config in configs
                }
                fit_answers, unit_frames = _breakout_readout_inputs(receipt)
                records = [
                    record
                    for record in transcript.records
                    if record["request_type"] == "BreakoutMethodMomentsRequest"
                    and record["request"]["breakout"] == identity
                ]
                by_metric = {}
                summary = []
                for config in configs:
                    matching = [
                        record
                        for record in records
                        if record["request"]["metric"]["name"] == config.metric.name
                    ]
                    assert matching
                    families = tuple(
                        (
                            record["request"]["method"]["name"],
                            tuple(_restore(record["answer"])["rows"]),
                        )
                        for record in matching
                    )
                    rows = families[0][1]
                    summary.extend(rows)
                    arms = frozenset(str(row["group_id"]) for row in rows)
                    digest, count = _digest_input_rows(rows)
                    by_metric[config.metric.name] = MetricEvidence(
                        rows=rows,
                        unit_frame=None,
                        observed_arms=arms,
                        n_treatment_arms=len(arms - {design.control_group}),
                        evidence_kind="moments_rows",
                        evidence_sha256=digest,
                        evidence_count=count,
                        decision_evidence_components=(("moments_rows", digest, count),),
                        cluster=scoped_context.cluster,
                        control_group=design.control_group,
                        method_rows=families,
                        method_fit_answers=fit_answers.get(config.metric.name, ()),
                    )
                options: dict[str, Any] = {
                    "metrics": tuple(config.metric for config in configs),
                    "control_group": design.control_group,
                    "dimension": identity["property"],
                    "source": stamp_source,
                    "methods_by_metric": methods_by_metric,
                    "method_roles_by_metric": roles,
                    "_prior_by_metric": {config.metric.name: config.prior for config in configs},
                    "alpha": plan.alpha,
                    "correction": policy.correction,
                    "q": policy.q or plan.q,
                    "policy_name": "compiled_plan",
                    "unit_frames_by_metric": unit_frames,
                    "covariates_by_metric": {
                        config.metric.name: config.covariates for config in configs
                    },
                }
                rows = _consume_once(
                    "breakout",
                    {"summary": summary, "method_inputs": records, "scope_inputs": receipt},
                    {
                        name: value
                        for name, value in options.items()
                        if name != "unit_frames_by_metric"
                    },
                    lambda summary=summary, by_metric=by_metric, options=options: run_breakout(
                        summary,
                        _method_evidence_by_metric=by_metric,
                        **options,
                    ),
                )
                assembled = _scope_breakout_inputs(transcript, rows, identity, stamp_source)
                results.extend(assembled)
                parts.append(assembled)
        expected = _capture()["scenarios"][entry.scenario_id][entry.ingress]["daily_values"]
        if expected:
            _assert_old_readout_rows(results, expected)
            _assert_old_assembled_metadata(parts, transcript)
        else:
            # These scopes own current input identity, not an archived segment estimate.
            assert parts and all(
                _old_daily_consumer(entry.scenario_id, entry.ingress) is None for _part in parts
            )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("All captured daily and as-of data representations and full resolved options")
def test_frozen_day_axis_consumers_match_complete_ordered_old_rows():
    from pydantic import TypeAdapter

    from increment.estimation.inference import Prior
    from tests.evidence_fixtures.transcript import _restore
    from tests.test_evidence_identity_parity import _capture

    for entry in scenario_roster():
        if entry.disposition != "source":
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        records = [r for r in transcript.records if r["request_type"] == "DayAxisRequest"]
        if not records:
            continue
        configuration, _context, design, _configs, plan = _consumer_context(transcript)
        results = []
        parts = []
        for record in records:
            request = _decode_transcript_request(record)
            options = configuration.get("day_axis_options")
            kwargs = None if options is None else _restore(options["kwargs"])
            configs = (
                () if options is None else tuple(_decode_config(c) for c in options["configs"])
            )
            if kwargs is not None:
                if kwargs.get("methods") is not None:
                    kwargs["methods"] = [
                        Method.model_validate(method) for method in kwargs["methods"]
                    ]
                kwargs["methods_by_metric"] = {
                    name: [Method.model_validate(method) for method in methods]
                    for name, methods in kwargs["methods_by_metric"].items()
                }
                kwargs["prior_by_metric"] = {
                    name: None if prior is None else TypeAdapter(Prior).validate_python(prior)
                    for name, prior in kwargs["prior_by_metric"].items()
                }
            assembled, scope_inputs = _consume_once(
                "day_axis",
                record["answer"],
                {
                    "request": request,
                    "plan": plan,
                    "design": design,
                    "lift_kwargs": kwargs,
                    "configs": configs,
                },
                lambda record=record, request=request, plan=plan, design=design, kwargs=kwargs, configs=configs: (
                    _day_axis_with_inputs(
                        record,
                        request,
                        plan=plan,
                        design=design,
                        lift_kwargs=kwargs,
                        configs=configs,
                    )
                ),
            )
            from tests.evidence_fixtures.transcript import _plain

            assert scope_inputs.keys() == {"identity_rows", "request", "plan", "source_components"}
            assert scope_inputs["request"] == record["answer"]["scope_request"]
            assert scope_inputs["plan"] == _plain(plan)
            assert scope_inputs["source_components"] == []
            _assert_common_snapshot(
                assembled,
                scope_inputs["identity_rows"],
                _restore(record["answer"]["scope_request"]),
                plan,
                route=record["answer"]["route"],
                view=request.grain,
            )
            _register_readout_identities(
                assembled, artifact_provenance=record["answer"]["artifact_provenance"]
            )
            results.extend(assembled)
            parts.append(assembled)
        expected = _capture()["scenarios"][entry.scenario_id][entry.ingress]["daily_values"]
        _assert_old_readout_rows(results, expected)
        _assert_old_assembled_metadata(parts, transcript)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("Every encouragement breakout input and full source-qualified row identity")
def test_frozen_encouragement_breakout_consumers_match_complete_ordered_old_rows():

    from increment.estimation.encouragement import ESTIMANDS
    from tests.evidence_fixtures.method_replay import (
        EncouragementReplayConfig,
        assert_encouragement_breakout_matches_old_row,
        estimate_encouragement_breakout_transcript,
    )
    from tests.test_evidence_identity_parity import _capture

    for entry in scenario_roster():
        if entry.disposition != "source" or entry.temporal:
            continue
        transcript = _frozen_coordinate(entry.scenario_id, entry.ingress)
        records = [
            record
            for record in transcript.records
            if record["request_type"] == "BreakoutMethodMomentsRequest"
        ]
        if not records:
            continue
        configuration, context, design, configs, plan = _consumer_context(transcript)
        if design.mechanism != "encouragement":
            continue
        if not any(scope["view"] == "breakout" for scope in configuration["scopes"]):
            continue
        requests = tuple(_decode_transcript_request(record) for record in records)
        methods = tuple(dict.fromkeys(request.method for request in requests))
        estimands = tuple(configuration["estimands"] or ESTIMANDS)
        options = EncouragementReplayConfig(
            design=design,
            metrics=tuple(config.metric for config in configs),
            estimands=estimands,
            methods=methods,
            prior=None,
            alpha=plan.alpha,
            alternative="two-sided",
            null_lift=None,
            null_abs=None,
            inference=None,
            cluster=context["cluster"],
            method_roles={
                method.name: "decision"
                if any(method == c.decision_method for c in configs)
                else "sensitivity"
                for method in methods
            },
            compliance_requested="compliance" in estimands,
            population=configuration["population"],
            as_of=None,
            completed_windows_only=False,
        )
        computed = _consume_once(
            "encouragement_breakout",
            records,
            options,
            lambda transcript=transcript, requests=requests, options=options: (
                estimate_encouragement_breakout_transcript(transcript, requests, options)
            ),
        )
        source_partitions = tuple(
            partition
            for record in transcript.records
            if record["request_type"] == "BreakoutSlicesRequest"
            for partition in _breakout_source_partitions(record)
        )
        expected = _capture()["scenarios"][entry.scenario_id][entry.ingress]["daily_values"]
        expected_by_key = {_full_readout_coordinate(row): row for row in expected}
        assert len(expected_by_key) == len(expected)
        observed = []
        projected_by_source = {}
        for request in requests:
            source = next(
                source
                for identity, source in source_partitions
                if identity
                == {
                    "property": request.breakout.property,
                    "source": request.breakout.source,
                }
            )
            counts = {}
            for row in transcript.answer_for(request)["rows"]:
                segment = (
                    "__null__"
                    if row[request.breakout.property] is None
                    else str(row[request.breakout.property])
                )
                counts.setdefault(segment, {})[str(row["group_id"])] = float(row["n"])
            for segment in sorted(value for member, value in computed if member == request):
                for result in computed[(request, segment)].results:
                    identity = (
                        request.metric.name,
                        result.group_id,
                        result.method,
                        result.estimand,
                        result.value_scale,
                        request.breakout.property,
                        segment,
                        source,
                        request.population,
                        None,
                    )
                    assert identity in expected_by_key and identity not in observed
                    projected = assert_encouragement_breakout_matches_old_row(
                        result,
                        expected_by_key[identity],
                        request=request,
                        segment=segment,
                        source=source,
                        n_treat=counts[segment].get(result.group_id),
                        n_control=counts[segment].get(design.control_group),
                    )
                    projected_by_source.setdefault(
                        (request.breakout.property, request.breakout.source, source),
                        [],
                    ).append(projected)
                    observed.append(identity)
        assert observed == [_full_readout_coordinate(row) for row in expected]
        complete = []
        parts = []
        for (dimension, declared_source, stamp_source), rows in projected_by_source.items():
            identity = {"property": dimension, "source": declared_source}
            assembled = _scope_breakout_inputs(transcript, rows, identity, stamp_source)
            parts.append(assembled)
            complete.extend(assembled)
        _assert_old_readout_rows(complete, expected, nominal_alpha=plan.alpha)
        _assert_old_assembled_metadata(parts, transcript)


@pytest.mark.integration
def test_source_only_capture_rejects_relabelled_summary_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evidence_fixtures import scenarios
    from tests.evidence_fixtures.scenarios import acquire_scenario
    from tests.parity_harness.cases import PARITY_CASES, _arm_facts_parity_case, _Fixture

    scenario_id = "arm-facts-parity"
    ingress = "from_unit_summary"
    replacement = _arm_facts_parity_case(fixture=_Fixture(unit_id_prefix="relabelled-"))
    monkeypatch.setattr(
        scenarios,
        "PARITY_CASES",
        tuple(replacement if case.id == scenario_id else case for case in PARITY_CASES),
    )
    frozen = _frozen_coordinate(scenario_id, ingress)
    live = acquire_scenario(scenario_id, ingress, baseline_revision=frozen.baseline_revision)
    from tests.test_evidence_identity_parity import _capture

    archived_identity = _capture()["scenarios"][scenario_id][ingress]["population_identity"]
    assert live.source_membership_identity is not None
    assert live.source_membership_identity != archived_identity
    for population, arms in archived_identity.items():
        assert {arm: evidence["unit_count"] for arm, evidence in arms.items()} == {
            arm: evidence["unit_count"]
            for arm, evidence in live.source_membership_identity[population].items()
        }

    with pytest.raises(AssertionError):
        _assert_coordinate_inputs(live, frozen)
