from __future__ import annotations

import json
import math
import warnings
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any, cast

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
def _canonicalize_denominator_support(value):
    if isinstance(value, list):
        return [_canonicalize_denominator_support(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                sorted(
                    (_canonicalize_denominator_support(item) for item in item_value),
                    key=str,
                )
                if key == "denominator_support"
                else _canonicalize_denominator_support(item_value)
            )
            for key, item_value in value.items()
        }
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            return json.dumps(
                _canonicalize_denominator_support(json.loads(value)),
                sort_keys=True,
                separators=(",", ":"),
            )
        except json.JSONDecodeError:
            return value
    return value


def _normalize_warehouse_replay_rows(rows):
    normalized = []
    for source_row in rows:
        row = {key: _canonicalize_denominator_support(value) for key, value in source_row.items()}
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


def test_warehouse_replay_comparison_ignores_denominator_support_order():
    rows = [
        {
            "evidence": '{"denominator_support":[{"arm":"control"},{"arm":"treatment"}]}',
            "retained_family": "[]",
        }
    ]
    reordered = [
        {
            "evidence": '{"denominator_support":[{"arm":"treatment"},{"arm":"control"}]}',
            "retained_family": "[]",
        }
    ]

    assert _normalize_warehouse_replay_rows(rows) == _normalize_warehouse_replay_rows(reordered)


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
        self.unit_frame_metrics = []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def unit_frame(self, metric, *args, **kwargs):
        self.unit_frame_reads += 1
        self.unit_frame_metrics.append(metric.name)
        return self.source.unit_frame(metric, *args, **kwargs)


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
            "x": [float(i % 3) for i in range(12)],
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


@pytest.mark.integration
def test_analysis_run_shares_one_snapshot_and_execution_for_multiple_metrics(
    seeded_pre_period_con, seeded_defs, monkeypatch
):
    from contextlib import contextmanager

    from increment import Method
    from tests.analysis_factory import held_source

    analysis = Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    source = held_source(analysis)
    source_type = type(source)
    snapshot_method = "readout_snapshot"
    execution_method = "readout_execution"
    original_snapshot = getattr(source_type, snapshot_method)
    original_execution = getattr(source_type, execution_method)
    snapshots = 0
    executions = 0

    @contextmanager
    def counted_snapshot(self, **kwargs):
        nonlocal snapshots
        snapshots += 1
        with original_snapshot(self, **kwargs) as pinned:
            yield pinned

    @contextmanager
    def counted_execution(self):
        nonlocal executions
        executions += 1
        with original_execution(self):
            yield

    monkeypatch.setattr(source_type, "readout_snapshot", counted_snapshot)
    monkeypatch.setattr(source_type, "readout_execution", counted_execution)
    try:
        rows = analysis.run(
            metrics=["avg_session_duration", "purchase_rate"],
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
        assert {row.metric for row in rows} == {"avg_session_duration", "purchase_rate"}
        assert snapshots == 1
        assert executions == 1
    finally:
        analysis.close()


def test_observational_readout_shares_frames_with_identity_digest():
    from collections import Counter

    from increment import AdjustmentSet
    from increment.estimation.engine import Method
    from increment.readouts import run
    from increment.semantics.design import Observational
    from tests.analysis_factory import held_source

    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(40)],
            "arm": ["C"] * 20 + ["T"] * 20,
            "x": [float(i % 5) for i in range(40)],
            "y": [float(i + 1) for i in range(40)],
            "z": [float((i * 3) % 11) for i in range(40)],
        }
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics={"y": "mean", "z": "mean"},
        design=design,
    )
    source = _CountingSource(cast(MomentSource, held_source(analysis)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = run(cast(MomentSource, source), decision_method=Method(name="iptw"))
        assert {row.metric for row in rows} == {"y", "z"}
        assert Counter(source.unit_frame_metrics) == Counter({"y": 1, "z": 1})
    finally:
        analysis.close()


def test_call_time_unadjusted_override_reads_no_moments_cube_unit_frames(tmp_path):
    from increment import AdjustmentSet
    from increment.estimation.engine import Method
    from increment.readouts import run
    from increment.semantics.design import Observational
    from tests.analysis_factory import held_source

    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(40)],
            "arm": ["C"] * 20 + ["T"] * 20,
            "x": [float(i % 5) for i in range(40)],
            "y": [float(i + 1) for i in range(40)],
        }
    )
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    spec = MetricSpec(name="y", type="mean", decision_method=Method(name="iptw"))
    original = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[spec],
        design=design,
    )
    path = tmp_path / "unadjusted_override.parquet"
    original.export(path)
    original.close()
    cube = Analysis.from_moments(pq.read_table(path).to_pylist(), metrics=[spec], design=design)
    source = _CountingSource(cast(MomentSource, held_source(cube)))
    try:
        rows = run(cast(MomentSource, source), decision_method=Method(name="unadjusted"))
        assert [row.method for row in rows] == ["unadjusted"]
        assert source.unit_frame_reads == 0
        assert source.unit_frame_metrics == []
    finally:
        cube.close()


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


def test_adjustment_kernel_runs_on_plain_frames():
    from increment import AdjustmentSet
    from increment.estimation.adjust import ADJUSTMENTS, estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import Observational

    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    analysis, source, metric = _counted_analysis(design=design)
    try:
        import sys

        facade = estimate_ate(source, design, methods=[Method(name="iptw")]).results
        input_type = sys.modules["increment.estimation._adjust.common"].AdjustmentInputs
        kernel = ADJUSTMENTS.get("iptw")(
            input_type(
                source.unit_frame(metric, covariates=["x"]),
                source.moments(metric),
                source.context.cluster,
            ),
            metric,
            design,
        )
        assert [row.model_dump() for row in kernel] == [row.model_dump() for row in facade]
    finally:
        analysis.close()


def test_adjusted_observational_methods_share_unit_frame_read():
    from increment import AdjustmentSet
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import Observational

    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    analysis, source, _metric = _counted_analysis(design=design)
    try:
        estimate_ate(source, design, methods=[Method(name="iptw"), Method(name="dml")])
        assert source.unit_frame_reads == 1
    finally:
        analysis.close()


def test_unadjusted_observational_estimator_reads_no_unclustered_unit_rows():
    from increment import AdjustmentSet
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import Observational

    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    analysis, source, _metric = _counted_analysis(design=design)
    try:
        estimate_ate(source, design, methods=[Method(name="unadjusted")])
        assert source.unit_frame_reads == 0
    finally:
        analysis.close()


def test_clustered_unadjusted_observational_estimator_reads_each_metric_once():
    from increment import AdjustmentSet
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import Observational

    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))
    analysis, source, _metric = _counted_analysis(cluster=True, design=design)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimate_ate(source, design, methods=[Method(name="unadjusted")])
        assert source.unit_frame_reads == 1
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


def test_adjustment_refusals_precede_unit_frame_reads():
    from increment import AdjustmentSet
    from increment.errors import CodedError
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import Observational

    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("x",)))

    cases = (
        ("null_lifts", "model.field.nonfinite"),
        ("alpha", "model.field.range"),
    )
    for parameter, expected_code in cases:
        analysis, source, _metric = _counted_analysis(cluster=True, design=design)
        try:
            with pytest.raises(CodedError) as raised:
                if parameter == "null_lifts":
                    estimate_ate(
                        cast(Any, source),
                        design,
                        methods=[Method(name="unadjusted")],
                        null_lifts={"y": math.inf},
                    )
                else:
                    estimate_ate(
                        cast(Any, source),
                        design,
                        methods=[Method(name="unadjusted")],
                        alpha=1.0,
                    )
            assert raised.value.code == expected_code
            assert source.unit_frame_reads == 0
        finally:
            analysis.close()

    allow_design = design.model_copy(
        update={"adjustment": AdjustmentSet(covariates=("x",), missing="allow")}
    )
    analysis, source, _metric = _counted_analysis(design=allow_design)
    try:
        with pytest.raises(CodedError) as raised:
            estimate_ate(cast(Any, source), allow_design, methods=[Method(name="iptw")])
        assert raised.value.code == "adjust.identification.allow_requires_learner"
        assert source.unit_frame_reads == 0
    finally:
        analysis.close()


def test_ratio_then_mean_moments_cube_warns_before_one_unit_frame_refusal(tmp_path):
    from increment.errors import CodedError
    from increment.estimation.adjust import estimate_ate
    from increment.estimation.engine import Method
    from increment.semantics.design import AdjustmentSet, Observational

    rows = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(24)],
            "arm": ["C"] * 12 + ["T"] * 12,
            "x": [float(i % 3) for i in range(24)],
            "revenue": [float(i + 1) for i in range(24)],
            "orders": [float(i % 4 + 1) for i in range(24)],
        }
    )
    specs = [
        MetricSpec(name="rpo", type="ratio", numerator="revenue", denominator="orders"),
        MetricSpec(name="revenue", type="mean"),
    ]
    design = Observational(
        control_group="C",
        adjustment=AdjustmentSet.model_construct(covariates=()),
    )
    original = Analysis.from_unit_summary(
        rows, unit="unit", group="arm", metrics=specs, design=design
    )
    path = tmp_path / "moments.parquet"
    original.export(path)
    original.close()
    cube = Analysis.from_moments(pq.read_table(path).to_pylist(), metrics=specs, design=design)
    from tests.analysis_factory import held_source

    cube_source = cast(MomentSource, held_source(cube))
    source = _CountingSource(cube_source)
    try:
        with pytest.warns(IncrementWarning) as caught:
            with pytest.raises(CodedError) as raised:
                estimate_ate(cast(Any, source), design, methods=[Method(name="iptw")])
        assert "estimation.adjust.skip_unsupported_metric" in {
            getattr(warning.message, "code", None) for warning in caught
        }
        assert raised.value.code == "evidence.unit_rows_unavailable"
        assert source.unit_frame_reads == 1
    finally:
        cube.close()
