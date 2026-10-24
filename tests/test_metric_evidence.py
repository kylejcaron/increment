from __future__ import annotations

import json
import math
import warnings
from datetime import date
from numbers import Real
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import Analysis, MetricSpec
from increment.errors import IncrementWarning
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan
from scripts.capture_evidence_identity import (
    _frame_rows,
    _rows,
    _views,
    build_analyses,
    replay_export,
)
from tests.parity_harness import dataset as ds
from tests.parity_harness.comparison import nested_close

_FIXTURES = Path(__file__).parent / "fixtures" / "evidence_identity"


def _strict_nested_close(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, Real) and isinstance(actual, Real):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        return math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(
            _strict_nested_close(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _strict_nested_close(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _assert_replay_rows_close(expected, actual):
    # Reuse the parity harness structure/null check, then pin floats more tightly.
    assert nested_close(expected, actual)
    assert _strict_nested_close(expected, actual), (
        "replay numeric differences exceed rtol=1e-12, atol=1e-15"
    )


def _without_platform_sensitive_identity_hashes(value):
    if isinstance(value, list):
        return [_without_platform_sensitive_identity_hashes(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _without_platform_sensitive_identity_hashes(item) for key, item in value.items()
    }
    components = value.get("evidence_components", []) + value.get(
        "decision_evidence_components", []
    )
    if any(component.get("kind", "").startswith("warehouse_qr:") for component in components):
        normalized["source_snapshot_id"] = None
    for key in ("evidence_components", "decision_evidence_components"):
        normalized[key] = [
            {
                field: item
                for field, item in component.items()
                if not (component.get("kind", "").startswith("warehouse_qr:") and field == "sha256")
            }
            for component in value.get(key, [])
        ]
    return normalized


def _assert_identity_equal(expected, actual):
    assert _without_platform_sensitive_identity_hashes(expected) == (
        _without_platform_sensitive_identity_hashes(actual)
    )


# QR floats differ at machine epsilon across BLAS platforms, making hashes platform-dependent;
# numerical fit payloads are tracked separately.
def _normalize_warehouse_replay_rows(rows):
    normalized = []
    for source_row in rows:
        row = dict(source_row)
        row.pop("source_snapshot_id", None)
        row.pop("family_id", None)
        retained = json.loads(row["retained_family"])
        for entry in retained:
            entry.pop("family_id", None)
            entry.pop("source_snapshot_id", None)
        retained.sort(
            key=lambda entry: json.dumps(
                {
                    "name": entry.get("name"),
                    "axes": (entry.get("family") or {}).get("axes"),
                    "members": entry.get("members"),
                },
                sort_keys=True,
            )
        )
        row["retained_family"] = json.dumps(retained, sort_keys=True, separators=(",", ":"))
        normalized.append(row)
    return normalized


def _warehouse_fit_payload(cube_path):
    rows = pq.read_table(cube_path).to_pylist()
    return [
        json.loads(row["cuped_evidence"])["evidence"]["fit_state"]
        for row in rows
        if row.get("record_kind") == "cuped_evidence" and row.get("cuped_evidence")
    ]


@pytest.mark.slow
@pytest.mark.integration
def test_identity_fixtures_unchanged(tmp_path):  # noqa: PLR0915
    fixture = json.loads((_FIXTURES / "readouts.json").read_text())
    analyses = build_analyses()
    try:
        baseline = analyses.pop("parity_baseline")
        _, baseline_identity = _views(baseline)
        original_results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IncrementWarning)
            original_results["parity_baseline"] = baseline.run()
            for name, analysis in analyses.items():
                original_results[name] = analysis.run()
        _assert_identity_equal(baseline_identity, fixture["parity_baseline"]["rows"])
        for name in analyses:
            _assert_identity_equal({"run": _rows(original_results[name])}, fixture[name]["rows"])

        # Analysis.export stamps format 13 and `_cuped_evidence.py` emits this
        # field with kind="fitted_multicovariate" for the retained fitted case.
        fitted_cube = pq.read_table(_FIXTURES / "exports" / "mixed_warehouse.parquet")
        assert fitted_cube.schema.metadata[b"increment.moments_format"] == b"13"
        assert any(
            json.loads(row["cuped_evidence"])["kind"] == "fitted_multicovariate"
            for row in fitted_cube.to_pylist()
            if row.get("record_kind") == "cuped_evidence"
        )
        live_cube = tmp_path / "mixed_warehouse.parquet"
        analyses["mixed_warehouse"].export(live_cube)
        # QR-derived fit floats can vary at machine epsilon across BLAS/LAPACK platforms.
        # Keep the BASE-captured numeric payload pinned within the replay tolerance.
        _assert_replay_rows_close(
            _warehouse_fit_payload(_FIXTURES / "exports" / "mixed_warehouse.parquet"),
            _warehouse_fit_payload(live_cube),
        )
        replay_templates = {
            "parity_baseline": baseline,
            "mixed_warehouse": analyses["mixed_warehouse"],
            "mixed_fitted_replay": analyses["mixed_fitted_replay"],
            "mixed_categorical_replay": analyses["mixed_categorical_replay"],
        }
        cube_names = {
            name: ("mixed_warehouse" if name == "mixed_fitted_replay" else name)
            for name in replay_templates
        }
        for name, original in replay_templates.items():
            payload = pq.read_table(
                _FIXTURES / "exports" / f"{cube_names[name]}.parquet"
            ).to_pylist()
            replay_rows = replay_export(original, payload)
            expected_rows = fixture[name]["replay_rows"]
            if name in {"mixed_warehouse", "mixed_fitted_replay"}:
                expected_rows = _normalize_warehouse_replay_rows(expected_rows)
                replay_rows = _normalize_warehouse_replay_rows(replay_rows)
            _assert_replay_rows_close(expected_rows, replay_rows)

        checkpoint_data = json.loads((_FIXTURES / "checkpoint.json").read_text())
        parent = Analysis.from_moments(
            checkpoint_data["checkpoint"],
            metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
            design=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            experiment_id="exp",
        )
        try:
            parent_snapshot = parent.sequential_snapshot()
            con = ds.duckdb_connection()
            panel = ds.sequential_unit_panel_frame(con)
            con.disconnect()
            as_of = date.fromisoformat(checkpoint_data["as_of"])
            prefix = panel.loc[panel["date"] <= as_of]
            later = pd.DataFrame(checkpoint_data["later_rows"])
            later["date"] = pd.to_datetime(later["date"]).dt.date
            later["exposure_date"] = pd.to_datetime(later["exposure_date"]).dt.date
            continued_panel = pd.concat([prefix, later], ignore_index=True)
            continued = Analysis.from_unit_panel(
                continued_panel,
                unit="user_id",
                group="variant",
                date="date",
                metrics=[MetricSpec(name="revenue", type="mean", missing="zero", window_days=1)],
                design=Randomized(
                    control_group="control", allocation={"control": 0.5, "treatment": 0.5}
                ),
                experiment_id="exp",
                exposure_date="exposure_date",
                plan=AnalysisPlan(
                    primary="revenue",
                    inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
                ),
            )
            try:
                later_as_of = max(later["date"])
                continuation = continued.capture_sequential(
                    finalized=True, previous=parent_snapshot, as_of=later_as_of
                )
                expected_continuation = checkpoint_data["continuation"]
                actual_snapshot = {
                    "registration_id": continuation.registration_id,
                    "prefix_id": continuation.prefix_id,
                    "parent_id": continuation.parent_id,
                    "ancestors": [item.model_dump(mode="json") for item in continuation.ancestors],
                    "records": [item.model_dump(mode="json") for item in continuation.records],
                    "states": [item.model_dump(mode="json") for item in continuation.states],
                    "as_of": later_as_of.isoformat(),
                }
                assert len(continuation.records) > len(parent_snapshot.records)
                assert sum(state.n for state in continuation.states) > sum(
                    state.n for state in parent_snapshot.states
                )
                assert actual_snapshot == {
                    key: expected_continuation[key] for key in actual_snapshot
                }
                _assert_replay_rows_close(
                    expected_continuation["rows"], _frame_rows(continued.run_asof())
                )
                continuation.verify_parent(parent_snapshot)
            finally:
                continued.close()
        finally:
            parent.close()
    finally:
        for analysis in analyses.values():
            analysis.close()
        baseline.close()
