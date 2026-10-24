from __future__ import annotations

import json
import math
import warnings
from datetime import date
from numbers import Real
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

from increment import Analysis, MetricSpec
from increment.errors import IncrementWarning
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan
from increment.sources import MomentSource
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
        {
            key: json.loads(row["cuped_evidence"])["evidence"][key]
            for key in ("fit_state", "inference_summary")
        }
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


class _CountingSource:
    def __init__(self, source: MomentSource):
        self.source = source
        self.unit_frame_reads = 0

    def __getattr__(self, name):
        return getattr(self.source, name)

    def unit_frame(self, *args, **kwargs):
        self.unit_frame_reads += 1
        return self.source.unit_frame(*args, **kwargs)


def _counted_analysis(*, quantile=False, winsor=False, cluster=False, design=None):
    from tests.analysis_factory import held_source

    spec = (
        MetricSpec(name="y", type="quantile", quantile=0.5)
        if quantile
        else MetricSpec(name="y", type="mean", winsorization={"upper_percentile": 0.75})
        if winsor
        else MetricSpec(name="y", type="mean")
    )
    frame = pd.DataFrame(
        {
            "unit": [f"c{i}" for i in range(6)] + [f"t{i}" for i in range(6)],
            "arm": ["C"] * 6 + ["T"] * 6,
            "store": [f"s{i}" for i in range(12)],
            "y": [float(i + 1) for i in range(12)],
        }
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="C" if design is None else None,
        metrics=[spec],
        cluster="store" if cluster else None,
        design=design,
    )
    source = cast(MomentSource, held_source(analysis))
    return analysis, _CountingSource(source), source.context.metrics[0]


def test_quantile_lift_from_rows_equals_facade():
    from increment.estimation.quantile import estimate_quantile_lift, quantile_lift_from_rows

    analysis, source, metric = _counted_analysis(quantile=True)
    try:
        facade = estimate_quantile_lift(source, metric, "C")
        rows = source.unit_frame(metric)
        row_results = quantile_lift_from_rows(rows, metric, "C", design=source.context.design)
        assert isinstance(row_results, list)
        assert [row.model_dump() for row in row_results] == [row.model_dump() for row in facade]
    finally:
        analysis.close()


def test_read_free_refusals_precede_unit_frame_reads(monkeypatch):
    from increment.errors import CodedError
    from increment.estimation.engine import Method
    from increment.estimation.quantile import estimate_quantile_lift
    from increment.estimation.winsor import raw_state_from_source

    for inference, methods, code in (
        (object(), None, "sequential.route.unsupported"),
        (None, [Method(name="cuped", variance_reduction="cuped")], "arm.metric.quantile_cuped"),
    ):
        analysis, source, metric = _counted_analysis(quantile=True)
        try:
            with pytest.raises(CodedError) as raised:
                if inference is not None:
                    estimate_quantile_lift(source, metric, "C", inference=inference)
                else:
                    estimate_quantile_lift(source, metric, "C", methods=methods)
            assert raised.value.code == code
            assert source.unit_frame_reads == 0
        finally:
            analysis.close()

    from tests.analysis_factory import held_source

    analysis, source, metric = _counted_analysis(winsor=True, cluster=True)
    try:
        with pytest.raises(CodedError) as raised:
            raw_state_from_source(source, metric)
        assert raised.value.code == "estimation.winsor.design_unsupported"
        assert source.unit_frame_reads == 0

        backing = cast(MomentSource, held_source(analysis))
        source_type = type(backing)
        original_unit_frame = source_type.unit_frame
        reads = 0

        def count_unit_frame(instance, *args, **kwargs):
            nonlocal reads
            reads += 1
            return original_unit_frame(instance, *args, **kwargs)

        monkeypatch.setattr(source_type, "unit_frame", count_unit_frame)
        with pytest.raises(CodedError) as raised:
            analysis.run()
        assert raised.value.code == "readout.metric.percentile_winsorization"
        assert reads == 0
    finally:
        analysis.close()


def test_refusal_precedes_any_loader_read():
    from increment import AdjustmentSet
    from increment.errors import CodedError
    from increment.estimation.quantile import estimate_quantile_lift
    from increment.semantics.design import Observational

    analysis, source, metric = _counted_analysis(
        quantile=True,
        design=Observational(control_group="C", adjustment=AdjustmentSet(covariates=("store",))),
    )
    try:
        with pytest.raises(CodedError) as raised:
            estimate_quantile_lift(source, metric, "C")
        assert raised.value.code == "readout.observational.quantile"
        assert source.unit_frame_reads == 0
    finally:
        analysis.close()


def test_quantile_rows_refuse_observational_design():
    from increment import AdjustmentSet
    from increment.errors import CodedError
    from increment.estimation.quantile import quantile_lift_from_rows
    from increment.semantics.design import Observational

    analysis, source, metric = _counted_analysis(
        quantile=True,
        design=Observational(control_group="C", adjustment=AdjustmentSet(covariates=("store",))),
    )
    try:
        with pytest.raises(CodedError) as raised:
            quantile_lift_from_rows(
                source.unit_frame(metric),
                metric,
                "C",
                design=source.context.design,
                authenticate=getattr(source, "validated_metric", None),
            )
        assert raised.value.code == "readout.observational.quantile"
    finally:
        analysis.close()
