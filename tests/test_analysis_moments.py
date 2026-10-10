"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import json
import warnings
from typing import cast

import pytest

from increment import Analysis
from increment.errors import CapabilityError, InvalidRequestError, WireFormatError
from increment.estimation.diagnostics import SRMResult
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.semantics.design import AdjustmentSet, Observational
from increment.semantics.loader import load
from increment.semantics.models import AnalysisPlan, MeanMetric, Winsorization
from tests.analysis_factory import make_analysis
from tests.sequential_cases import registration


def _lift_rows(rows: object) -> list[LiftEstimate]:
    return cast(list[LiftEstimate], rows)


def _analysis_with_metrics(con, definitions_path, metrics, *, store):
    """Build a native source whose experiment plan names the selected metrics."""
    defs = load(definitions_path)
    experiment = defs.experiment("new_onboarding_v2")
    assert experiment is not None
    metric_names = [metric.name for metric in metrics]
    experiment = experiment.model_copy(update={"plan": AnalysisPlan(primary=metric_names)})
    defs = defs.model_copy(
        update={
            "metrics": [*defs.metrics, *metrics],
            "experiments": [
                experiment if candidate.name == experiment.name else candidate
                for candidate in defs.experiments
            ],
        }
    )
    return make_analysis(
        con,
        defs,
        experiment=experiment,
        metrics=list(metrics),
        store=store,
    )


def _moment_rows(**extra: object) -> list[dict[str, object]]:
    base = {
        "experiment_id": "e1",
        "metric": "revenue",
        "n": 2,
        "successes": None,
        "ref_x": None,
        "cx1": None,
        "cx2": None,
        "cxy": None,
        "ref_den": None,
        "cden1": None,
        "cden2": None,
        "cyden": None,
        "x_role": None,
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": None,
        "winsor_lower_bound": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
        "moments_format": 10,
        **extra,
    }
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    base["decision_plan"] = compiled_plan_to_json(
        compile_decision_plan(
            None,
            [MeanMetric(name="revenue", entity="user", fact="revenue")],
        )
    )
    return [
        {**base, "group_id": "treatment", "ref_y": 30.0, "cy1": 0.0, "cy2": 50.0},
        {**base, "group_id": "control", "ref_y": 17.75, "cy1": 0.0, "cy2": 44.5},
    ]


def test_current_fixed_horizon_moments_round_trip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "arm": ["control", "control", "treatment", "treatment"],
                "revenue": [10.0, 12.0, 18.0, 20.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean"},
        control="control",
    )
    path = tmp_path / "current.parquet"
    original.export(path)
    payload = pq.read_table(path).to_pylist()
    assert {row["moments_format"] for row in payload} == {11}
    replay = Analysis.from_moments(payload, metrics={"revenue": "mean"}, control="control")
    (original_row,) = original.run()
    (replay_row,) = replay.run()
    assert replay_row.model_dump(exclude={"source_snapshot_id"}) == original_row.model_dump(
        exclude={"source_snapshot_id"}
    )
    # Format-11 moments retain the exact source identity and request semantics,
    # so content identity is shared across the live and replayed source.
    assert replay_row.source_snapshot_id == original_row.source_snapshot_id


def test_legacy_moments_identity_is_stable_across_reload(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [1, 2, 3, 4],
                "arm": ["control", "control", "treatment", "treatment"],
                "revenue": [10.0, 12.0, 18.0, 20.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean"},
        control="control",
        experiment_id="exp",
    )
    source_path = tmp_path / "source.parquet"
    original.export(source_path)
    legacy_rows = [
        {key: value for key, value in row.items() if key != "source_identity"}
        | {"moments_format": 10}
        for row in pq.read_table(source_path).to_pylist()
    ]
    legacy_path = tmp_path / "legacy.parquet"
    pq.write_table(pa.Table.from_pylist(legacy_rows), legacy_path)
    legacy = Analysis.from_moments(
        pq.read_table(legacy_path).to_pylist(),
        metrics={"revenue": "mean"},
        control="control",
        experiment_id="exp",
    )
    reloaded = None
    try:
        (original_row,) = original.run()
        (legacy_row,) = legacy.run()
        assert legacy_row.source_snapshot_id != original_row.source_snapshot_id

        reloaded = Analysis.from_moments(
            pq.read_table(legacy_path).to_pylist(),
            metrics={"revenue": "mean"},
            control="control",
            experiment_id="exp",
        )
        (reloaded_row,) = reloaded.run()
        assert reloaded_row.source_snapshot_id == legacy_row.source_snapshot_id
    finally:
        if reloaded is not None:
            reloaded.close()
        legacy.close()
        original.close()


def test_triggered_run_refuses_unit_summary_and_moment_replay():
    import pyarrow as pa

    analysis = Analysis.from_unit_summary(
        pa.table({"unit": [1, 2], "arm": ["control", "treatment"], "revenue": [1.0, 2.0]}),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean"},
        control="control",
    )
    replay = Analysis.from_moments(_moment_rows(), metrics={"revenue": "mean"}, control="control")
    panel = Analysis.from_unit_panel(
        pa.table(
            {
                "unit": [1, 2],
                "arm": ["control", "treatment"],
                "day": ["2025-01-01", "2025-01-01"],
                "revenue": [1.0, 2.0],
            }
        ),
        unit="unit",
        group="arm",
        date="day",
        metrics={"revenue": "mean"},
        control="control",
    )
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    switchback = Analysis.from_switchback_panel(
        pa.Table.from_pylist(
            [
                {
                    "unit": unit,
                    "cycle": cycle,
                    "period": period,
                    "step": step,
                    "group": group,
                    "value": float(cycle + period + step),
                }
                for unit, order in (
                    ("u1", ("control", "treatment")),
                    ("u2", ("treatment", "control")),
                )
                for cycle in range(2)
                for period, group in enumerate(order)
                for step in range(2)
            ]
        ),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={"value": UnitCycleTApproximation()},
    )
    candidates = (analysis, replay, panel, switchback)
    for candidate in candidates:
        with pytest.raises(CapabilityError) as refused:
            candidate.run(population="triggered")
        assert refused.value.code == "facade.analysis.trigger_unsupported"
        assert refused.value.context["route"] in {"moments", "native", "switchback"}
        assert refused.value.context["supported_sources"] == (
            "from_definitions",
            "from_unit_day_artifact",
        )
        assert "Analysis.from_definitions" in str(refused.value)
        assert "Analysis.from_unit_day_artifact" in str(refused.value)


@pytest.mark.parametrize("version", [7, 8])
def test_legacy_moments_cannot_claim_exact_count_transport(version):
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(
            _moment_rows(moments_format=version), metrics={"revenue": "mean"}, control="control"
        )
    assert exc.value.code == "moments.format.unsupported_legacy"


def test_moments_reject_noncanonical_source_identity_at_replay():
    from increment.sources import SOURCE_IDENTITY_FIELD

    rows = _moment_rows()
    for row in rows:
        row[SOURCE_IDENTITY_FIELD] = json.dumps({"oversized": 2**53})

    with pytest.raises(WireFormatError) as refused:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert refused.value.code == "moments.format.invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [("n", float(2**61)), ("successes", float(2**61 - 513)), ("successes", True)],
)
def test_moments_refuse_counts_encoded_as_floats_or_booleans(field, value):
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(
            _moment_rows(**{field: value}), metrics={"revenue": "mean"}, control="control"
        )
    assert exc.value.code == "moments.count_not_integer"


def test_current_moments_require_the_nullable_success_count_field():
    rows = _moment_rows()
    del rows[0]["successes"]
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert exc.value.code == "moments.count_field_missing"


@pytest.mark.parametrize(
    ("field", "value"), [("n", 0), ("n", -1), ("successes", -1), ("successes", 3)]
)
@pytest.mark.parametrize("selected", [True, False])
def test_moments_count_ranges_are_validated_even_for_unselected_rows(field, value, selected):
    rows = _moment_rows()
    if selected:
        rows[0][field] = value
    else:
        rows.append({**rows[0], "metric": "unselected", field: value})
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert exc.value.code == "moments.count_out_of_range"
    assert exc.value.context["field"] == field


@pytest.mark.parametrize(
    ("field", "value"), [("n", 0), ("n", -1), ("successes", -1), ("successes", 3)]
)
def test_export_refuses_invalid_provider_counts_before_writing(field, value, monkeypatch, tmp_path):
    import json

    from increment._frame_validation import refuse_observational_quantile
    from increment.sources import ASSIGNMENT_COUNTS_FIELD, MomentsSource, export_source_moments

    source = MomentsSource(
        _moment_rows(**{ASSIGNMENT_COUNTS_FIELD: json.dumps({"control": 2, "treatment": 2})}),
        metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
        study_id="e1",
    )
    moments = source.moments

    def invalid_moments(metric):
        rows = [dict(row) for row in moments(metric)]
        rows[0][field] = value
        return rows

    monkeypatch.setattr(source, "moments", invalid_moments)
    path = tmp_path / "invalid.parquet"
    with pytest.raises(WireFormatError) as exc:
        export_source_moments(source, path, observational_refusal=refuse_observational_quantile)
    assert exc.value.code == "moments.count_out_of_range"
    assert exc.value.context["field"] == field
    assert not path.exists()


@pytest.mark.slow
def test_export_round_trips_through_from_moments(seeded_con, seeded_defs, tmp_path):
    """export -> parquet -> from_moments reproduces run() exactly.

    Scoped to purchase_rate (conversion) and avg_session_duration (mean).
    """
    definitions = load(seeded_defs)
    experiment = definitions.experiment("new_onboarding_v2")
    assert experiment is not None
    design = experiment.resolved_design().model_copy(update={"allocation_scheme": "independent"})
    a = make_analysis(seeded_con, definitions, experiment=experiment, _design=design, store="auto")
    baseline = {(e.metric, e.group_id): e.require_lift().value for e in _lift_rows(a.run())}
    assert {"purchase_rate", "avg_session_duration"} <= {m for m, _ in baseline}

    p = tmp_path / "moments.parquet"
    a.export(p)
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.schema.metadata[b"increment.moments_format"] == b"11"
    rows = table.to_pylist()
    assert {r["moments_format"] for r in rows} == {11}
    assert {r["metric"] for r in rows} == {"purchase_rate", "avg_session_duration", "d7_retention"}

    b = Analysis.from_moments(
        rows,
        metrics={"purchase_rate": "conversion", "avg_session_duration": "mean"},
        design=design,
    )
    from increment.sources import ASSIGNMENT_COUNTS_FIELD

    payloads = {r[ASSIGNMENT_COUNTS_FIELD] for r in rows}
    assert len(payloads) == 1
    native_srm = a.srm(expected={"control": 0.5, "treatment": 0.5})
    restored_srm = b.srm(expected={"control": 0.5, "treatment": 0.5})
    assert isinstance(restored_srm, SRMResult)
    assert isinstance(native_srm, SRMResult)
    assert restored_srm.observed == native_srm.observed
    got = {(e.metric, e.group_id): e.require_lift().value for e in _lift_rows(b.run())}

    for key in (("purchase_rate", "treatment"), ("avg_session_duration", "treatment")):
        assert got[key] == pytest.approx(baseline[key], rel=1e-12), key


def test_percentile_winsorization_export_reloads_resolved_bounds(con, tmp_path):
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(upper_percentile=0.99),
    )
    analysis = _analysis_with_metrics(
        con,
        "examples/definitions",
        [metric],
        store="none",
    )

    path = tmp_path / "winsorized-moments.parquet"
    analysis.export(path)
    import pyarrow.parquet as pq

    _reloaded = Analysis.from_moments(
        pq.read_table(path).to_pylist(),
        metrics=[
            MetricSpec(
                name="revenue",
                winsorization={"upper_percentile": 0.99},
            )
        ],
        control="control",
    )
    exported_rows = pq.read_table(path).to_pylist()
    assert {row["winsor_upper_percentile"] for row in exported_rows} == {0.99}
    assert all(row["winsor_upper_bound"] is not None for row in exported_rows)

    expected_bound = exported_rows[0]["winsor_upper_bound"]
    assert expected_bound is not None
    fixed_rows = [
        {
            **row,
            "n": 200,
            "cy2": float(row["cy2"]) * 100,
            "winsor_upper_percentile": None,
            "winsor_n": 200,
            "winsor_n_upper": 200,
        }
        for row in pq.read_table(path).to_pylist()
    ]
    reloaded_output = Analysis.from_moments(
        fixed_rows,
        metrics=[
            MetricSpec(
                name="revenue",
                winsorization={"upper_value": expected_bound},
            )
        ],
        control="control",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        output_rows = _lift_rows(reloaded_output.run())
    assert output_rows
    assert all(row.winsor_upper_bound == pytest.approx(expected_bound) for row in output_rows)
    with pytest.raises(CapabilityError) as exc_info:
        _reloaded.run()
    assert exc_info.value.code == "estimation.winsor.raw_state_required"


def test_portable_subset_validates_only_selected_winsorization_declarations(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 30
    frame = pa.table(
        {
            "unit": list(range(2 * n)),
            "arm": ["control"] * n + ["treatment"] * n,
            "plain": [float(10 + i % 5) for i in range(2 * n)],
            "capped": [float(20 + i % 8) for i in range(2 * n)],
        }
    )
    metrics = [
        MetricSpec(name="plain"),
        MetricSpec(name="capped", winsorization={"upper_value": 25.0}),
    ]
    original = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=metrics,
    )
    path = tmp_path / "portable-winsorization.parquet"
    original.export(path)
    rows = pq.read_table(path).to_pylist()

    full = Analysis.from_moments(rows, control="control", metrics=metrics)
    subset = Analysis.from_moments(rows, control="control", metrics={"plain": "mean"})
    full_plain = {
        result.group_id: result.require_lift()
        for result in _lift_rows(full.run())
        if result.metric == "plain"
    }
    subset_plain = {
        result.group_id: result.require_lift()
        for result in _lift_rows(subset.run())
        if result.metric == "plain"
    }
    assert set(subset_plain) == set(full_plain) == {"treatment"}
    for group, observed in subset_plain.items():
        expected = full_plain[group]
        assert (observed.value, observed.lb, observed.ub) == pytest.approx(
            (expected.value, expected.lb, expected.ub), rel=1e-12
        )
    assert subset_plain["treatment"].value == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(CapabilityError) as missing:
        Analysis.from_moments(rows, control="control", metrics={"capped": "mean"})
    assert missing.value.code == "moments.winsorization.metadata_missing"

    malformed = [dict(row) for row in rows]
    capped_row = next(row for row in malformed if row["metric"] == "capped")
    capped_row["moments_format"] = 7
    with pytest.raises(WireFormatError) as global_refusal:
        Analysis.from_moments(malformed, control="control", metrics={"plain": "mean"})
    assert global_refusal.value.code == "moments.format.mixed"


def test_mixed_winsorization_metrics_survive_run_and_export(tmp_path):
    """A fixed-value-winsorized metric and an unwinsorized one must both
    survive run() and export(), despite differing winsor-percentile dtypes."""
    from datetime import datetime

    import ibis

    from tests.analysis_factory import _make_event_log_table

    winsorized = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(upper_value=1_000.0),
    )
    plain = MeanMetric(
        name="revenue_unwinsorized",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
    )
    # `con` (the shared micro fixture) is too thin for infer_lift's guards on
    # any relative-lift estimate; build a bigger-but-still-local warehouse so
    # run()/export() exercise the real end-to-end path without the slow
    # seeded warehouse.
    extra_rows = []
    for i in range(15):
        for arm, group in (("c", "control"), ("t", "treatment")):
            uid = f"extra_{arm}{i}"
            extra_rows.append(
                {
                    "event_at": datetime(2025, 1, 16, 9, 0, 0),
                    "user_id": uid,
                    "session_id": f"s_{uid}",
                    "event": "page_view",
                    "experiment_id": "new_onboarding_v2",
                    "group_id": group,
                    "revenue": None,
                    "duration_s": None,
                    "country_code": "US",
                    "device_type": "web",
                    "plan": "free",
                }
            )
            extra_rows.append(
                {
                    "event_at": datetime(2025, 1, 17, 9, 0, 0),
                    "user_id": uid,
                    "session_id": f"s_{uid}",
                    "event": "purchase",
                    "experiment_id": None,
                    "group_id": None,
                    "revenue": (10.0 + i) if arm == "c" else (12.0 + i * 1.1),
                    "duration_s": None,
                    "country_code": "US",
                    "device_type": "web",
                    "plan": "free",
                }
            )
    con = ibis.duckdb.connect()
    _make_event_log_table(con, extra_rows=extra_rows)

    analysis = _analysis_with_metrics(
        con,
        "examples/definitions",
        [winsorized, plain],
        store="none",
    )

    results = analysis.run()
    assert {e.metric for e in _lift_rows(results)} == {"revenue", "revenue_unwinsorized"}

    path = tmp_path / "mixed-winsorization-moments.parquet"
    analysis.export(path)
    import pyarrow.parquet as pq

    exported = pq.read_table(path).to_pylist()
    assert {row["metric"] for row in exported} == {"revenue", "revenue_unwinsorized"}


def test_from_unit_panel_control_and_design_mutually_exclusive():
    import pandas as pd

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "variant": ["treatment", "treatment", "control", "control"],
            "day": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "revenue": [10.0, 20.0, 5.0, 7.0],
        }
    )
    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("revenue",))
    )
    with pytest.raises(InvalidRequestError) as both:
        Analysis.from_unit_panel(
            df,
            unit="user_id",
            group="variant",
            date="day",
            control="control",
            metrics={"revenue": "mean"},
            design=design,
        )
    with pytest.raises(InvalidRequestError) as neither:
        Analysis.from_unit_panel(
            df, unit="user_id", group="variant", date="day", metrics={"revenue": "mean"}
        )
    assert both.value.code == neither.value.code == "query.fact_resolution.pass_exactly_one"


def test_from_moments_control_and_design_mutually_exclusive():
    rows = [
        {
            "experiment_id": "e",
            "metric": "revenue",
            "group_id": g,
            "n": 10,
            "sum_y": s,
            "sum_y2": q,
        }
        for g, s, q in (("T", 45.0, 240.0), ("C", 30.0, 110.0))
    ]
    design = Observational(control_group="C", adjustment=AdjustmentSet(covariates=("sum_y",)))
    with pytest.raises(InvalidRequestError) as both:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="C", design=design)
    with pytest.raises(InvalidRequestError) as neither:
        Analysis.from_moments(rows, metrics={"revenue": "mean"})
    assert both.value.code == neither.value.code == "query.fact_resolution.pass_exactly_one"


def test_from_moments_run_refuses_call_time_epistemic_policy_kwargs():
    """The frame/moments seam has no execute() dispatch to forward
    alpha=/alternative=/inference=/margins=/null_lifts=/margins_abs= to.
    Analysis.run() used to silently ignore every one of these on this
    path (e.g. margins={'revenue': 0.02} returned a result
    byte-identical to omitting it entirely - a non-inferiority
    guardrail silently degrading to a test against 0, with no error).
    Each must now raise TypeError instead of no-oping."""
    from increment import AlwaysValid

    rows = _moment_rows()
    a = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    baseline = _lift_rows(a.run())
    assert baseline[0].require_lift().level == pytest.approx(0.95)

    with pytest.raises(TypeError):
        a.run(alpha=0.01)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(alternative="greater")  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(inference=AlwaysValid(registration=registration("gaussian")))  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(margins={"revenue": 0.02})  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(null_lifts={"revenue": 0.02})  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        a.run(margins_abs={"revenue": 0.02})  # ty: ignore[unknown-argument]


def test_from_moments_declared_plan_reaches_role_based_dispatch():
    """`Analysis.from_moments(..., plan=)` didn't exist before this fix --
    the classmethod accepted no `plan=` at all, despite `MomentsSource`
    (its underlying source) already supporting one, and `run()`'s own
    refusal message told a seam caller to declare policy "via the
    source's plan= construction argument" even though no `from_*` seam
    constructor actually exposed one -- a caller following that advice
    hit a second, more confusing `TypeError: unexpected keyword argument
    'plan'`. Proves `plan=` is now wired all the way through to
    `readouts.run()`'s role-based dispatch, not just accepted and
    dropped: a declared `primary="revenue"` stamps `role="primary"` and
    `discovery=None`, while the same rows with no `plan=` stay
    undeclared (`role=None`)."""
    rows = _moment_rows()
    undeclared = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert undeclared.run()[0].role is None

    declared = Analysis.from_moments(
        rows,
        metrics={"revenue": "mean"},
        control="control",
        plan=AnalysisPlan(primary="revenue"),
    )
    result = _lift_rows(declared.run())[0]
    assert result.role == "primary"


def _v7_rows(experiment_id, ref_y_treatment, cy2):
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    plan = compiled_plan_to_json(
        compile_decision_plan(None, [MeanMetric(name="revenue", entity="user", fact="revenue")])
    )
    base = {
        "experiment_id": experiment_id,
        "metric": "revenue",
        "n": 200,
        "ref_x": None,
        "cx1": None,
        "cx2": None,
        "cxy": None,
        "ref_den": None,
        "cden1": None,
        "cden2": None,
        "cyden": None,
        "x_role": None,
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": None,
        "winsor_lower_bound": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
        "successes": None,
        "moments_format": 10,
        "decision_plan": plan,
    }
    return [
        {**base, "group_id": "treatment", "ref_y": ref_y_treatment, "cy1": 0.0, "cy2": cy2},
        {**base, "group_id": "control", "ref_y": 17.75, "cy1": 0.0, "cy2": 4450.0},
    ]


def test_moments_cube_refuses_duplicate_and_cross_experiment_arm_rows():
    """One cube is one experiment with exactly one row per (metric, group_id)."""
    import pytest

    from increment import Analysis
    from increment.errors import CodedError

    def run(rows):
        analysis = Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
        return [(row.group_id, row.require_lift().value) for row in _lift_rows(analysis.run())]

    rows = _v7_rows("e1", 30.0, 5000.0)
    extra = {**rows[0], "ref_y": 99.0, "n": 5, "cy2": 10.0}

    with pytest.raises(CodedError) as excinfo:
        run([*rows, extra])
    assert excinfo.value.code == "moments.rows.duplicate"
    with pytest.raises(CodedError) as cross_experiment:
        run([*rows, *_v7_rows("e2", 99.0, 10.0)])
    assert cross_experiment.value.code == "moments.rows.experiment_conflict"

    def answer(cube):
        try:
            return run(cube)
        except CodedError as exc:
            return exc.code

    assert answer([*rows, extra]) == answer([extra, *rows])


def test_moments_cube_refuses_disjoint_arms_from_different_experiments():
    import pytest

    from increment import Analysis
    from increment.errors import CodedError

    rows = [_v7_rows("e1", 30.0, 5000.0)[1], _v7_rows("e2", 99.0, 10.0)[0]]
    for cube in (rows, list(reversed(rows))):
        with pytest.raises(CodedError) as refusal:
            Analysis.from_moments(cube, metrics={"revenue": "mean"}, control="control")
        assert refusal.value.code == "moments.rows.experiment_conflict"
        assert refusal.value.context["experiments"] == ("e1", "e2")


def test_moments_cube_normalizes_arm_identity_before_duplicate_check():
    import pytest

    from increment import Analysis
    from increment.errors import CodedError

    rows = _v7_rows("e1", 30.0, 5000.0)
    rows[0]["group_id"] = 1
    extra = {**rows[0], "group_id": "1"}
    for cube in ([*rows, extra], [extra, *rows]):
        with pytest.raises(CodedError) as refusal:
            Analysis.from_moments(cube, metrics={"revenue": "mean"}, control="control")
        assert refusal.value.code == "moments.rows.duplicate"
