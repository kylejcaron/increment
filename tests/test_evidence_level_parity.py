from __future__ import annotations

import json
import warnings
from dataclasses import replace
from datetime import datetime

import pyarrow as pa
import pytest

from increment import Method, MetricSpec, Scope, from_frame
from increment.errors import IncrementWarning
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
        changed = EvidenceRecord([(key, {"moments": {"ref_x": -10.0}})])
        with pytest.raises(AssertionError):
            assert_same_evidence(EvidenceRecord([(key, record.entries[key])]), changed)


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
