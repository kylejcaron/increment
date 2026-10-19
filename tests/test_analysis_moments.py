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
from tests.capability_consumers_auxiliary import assert_refusal
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


def _legacy_fixed10_rows(**extra: object) -> list[dict[str, object]]:
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


def test_current_conformance_runs_restored_writer_rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": [1, 2, 3, 4, 5, 6],
                "arm": ["control", "control", "control", "treatment", "treatment", "treatment"],
                "revenue": [10.0, 12.0, 11.0, 18.0, 20.0, 19.0],
                "converted": [0, 1, 0, 1, 1, 0],
            }
        ),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean", "converted": "conversion"},
        control="control",
    )
    path = tmp_path / "current.parquet"
    original.export(path)
    payload = pq.read_table(path).to_pylist()
    assert {row["moments_format"] for row in payload} == {12}
    replay = Analysis.from_moments(
        payload, metrics={"revenue": "mean", "converted": "conversion"}, control="control"
    )
    original_rows = {row.metric: row for row in _lift_rows(original.run())}
    replay_rows = {row.metric: row for row in _lift_rows(replay.run())}
    assert replay_rows.keys() == original_rows.keys()
    for metric, original_row in original_rows.items():
        replay_row = replay_rows[metric]
        assert replay_row.value_scale == original_row.value_scale
        assert replay_row.require_lift().value == pytest.approx(
            original_row.require_lift().value, rel=1e-12
        )
        assert replay_row.require_lift().lb == pytest.approx(
            original_row.require_lift().lb, rel=1e-12
        )
        assert replay_row.require_lift().ub == pytest.approx(
            original_row.require_lift().ub, rel=1e-12
        )
        assert replay_row.source_snapshot_id == original_row.source_snapshot_id
        assert replay_row.model_dump(exclude={"source_snapshot_id"}) == original_row.model_dump(
            exclude={"source_snapshot_id"}
        )


def test_legacy_moments_preserve_missing_evidence():

    legacy = Analysis.from_moments(
        _legacy_fixed10_rows(),
        metrics={"revenue": "mean"},
        control="control",
    )
    result = _lift_rows(legacy.run())[0]
    assert result.require_lift().value == pytest.approx(0.6901408450703352)
    assert result.source_snapshot_id
    with pytest.raises(CapabilityError) as exc:
        legacy.srm(inference="fixed")
    assert exc.value.code == "source.moments.assignment_counts"


def test_legacy_fixed10_identity_is_stable_across_reload(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    legacy_rows = _legacy_fixed10_rows()
    assert {row["moments_format"] for row in legacy_rows} == {10}
    assert all("source_identity" not in row for row in legacy_rows)
    legacy_path = tmp_path / "legacy-fixed10.parquet"
    pq.write_table(pa.Table.from_pylist(legacy_rows), legacy_path)
    legacy = Analysis.from_moments(
        pq.read_table(legacy_path).to_pylist(),
        metrics={"revenue": "mean"},
        control="control",
    )
    reloaded = None
    try:
        (legacy_row,) = legacy.run()
        reloaded = Analysis.from_moments(
            pq.read_table(legacy_path).to_pylist(),
            metrics={"revenue": "mean"},
            control="control",
        )
        (reloaded_row,) = reloaded.run()
        assert reloaded_row.source_snapshot_id == legacy_row.source_snapshot_id
    finally:
        if reloaded is not None:
            reloaded.close()
        legacy.close()


def test_triggered_run_refuses_unit_summary_and_moment_replay():
    import pyarrow as pa

    analysis = Analysis.from_unit_summary(
        pa.table({"unit": [1, 2], "arm": ["control", "treatment"], "revenue": [1.0, 2.0]}),
        unit="unit",
        group="arm",
        metrics={"revenue": "mean"},
        control="control",
    )
    replay = Analysis.from_moments(
        _legacy_fixed10_rows(), metrics={"revenue": "mean"}, control="control"
    )
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
    candidates = (
        ("from_unit_summary", analysis),
        ("from_moments", replay),
        ("from_unit_panel", panel),
        ("from_switchback_panel", switchback),
    )
    for ingress, candidate in candidates:
        with pytest.raises(CapabilityError) as refused:
            candidate.run(population="triggered")
        assert refused.value.code == "facade.analysis.trigger_unsupported"
        assert refused.value.context["route"] in {"moments", "native", "switchback"}
        assert refused.value.context["supported_sources"] == (
            "from_definitions",
            "from_unit_day_artifact",
        )
        assert_refusal("triggered-sequential-boundary", ingress, refused.value, stage="run")


@pytest.mark.parametrize("version", [7, 8])
def test_legacy_moments_cannot_claim_exact_count_transport(version):
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(
            _legacy_fixed10_rows(moments_format=version),
            metrics={"revenue": "mean"},
            control="control",
        )
    assert exc.value.code == "moments.format.unsupported_legacy"


def test_moments_reject_noncanonical_source_identity_at_replay():
    from increment.sources import SOURCE_IDENTITY_FIELD

    rows = _legacy_fixed10_rows()
    for row in rows:
        row[SOURCE_IDENTITY_FIELD] = json.dumps({"oversized": 2**53})

    with pytest.raises(WireFormatError) as refused:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert refused.value.code == "moments.format.invalid"


def test_moments_reject_conflicting_boolean_and_integer_identities():
    from increment.sources import SOURCE_IDENTITY_FIELD

    rows = _legacy_fixed10_rows()
    rows[0][SOURCE_IDENTITY_FIELD] = json.dumps({"generation": True})
    rows[1][SOURCE_IDENTITY_FIELD] = json.dumps({"generation": 1})

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
            _legacy_fixed10_rows(**{field: value}),
            metrics={"revenue": "mean"},
            control="control",
        )
    assert exc.value.code == "moments.count_not_integer"


def test_current_moments_require_the_nullable_success_count_field():
    rows = _legacy_fixed10_rows()
    del rows[0]["successes"]
    with pytest.raises(WireFormatError) as exc:
        Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control")
    assert exc.value.code == "moments.count_field_missing"


@pytest.mark.parametrize(
    ("field", "value"), [("n", 0), ("n", -1), ("successes", -1), ("successes", 3)]
)
@pytest.mark.parametrize("selected", [True, False])
def test_moments_count_ranges_are_validated_even_for_unselected_rows(field, value, selected):
    rows = _legacy_fixed10_rows()
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
        _legacy_fixed10_rows(
            **{ASSIGNMENT_COUNTS_FIELD: json.dumps({"control": 2, "treatment": 2})}
        ),
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
    from increment.estimation.engine import Method

    definitions = load(seeded_defs)
    experiment = definitions.experiment("new_onboarding_v2")
    assert experiment is not None
    design = experiment.resolved_design().model_copy(update={"allocation_scheme": "independent"})
    a = make_analysis(seeded_con, definitions, experiment=experiment, _design=design, store="auto")
    baseline = {
        (e.metric, e.group_id): e.require_lift().value
        for e in _lift_rows(a.run(decision_method=Method(name="unadjusted")))
    }
    assert {"purchase_rate", "avg_session_duration"} <= {m for m, _ in baseline}

    p = tmp_path / "moments.parquet"
    a.export(p)
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.schema.metadata[b"increment.moments_format"] == b"12"
    rows = table.to_pylist()
    assert {r["moments_format"] for r in rows} == {12}
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
    got = {
        (e.metric, e.group_id): e.require_lift().value
        for e in _lift_rows(b.run(decision_method=Method(name="unadjusted")))
    }

    for key in (("purchase_rate", "treatment"), ("avg_session_duration", "treatment")):
        assert got[key] == pytest.approx(baseline[key], rel=1e-12), key


@pytest.mark.integration
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


@pytest.mark.integration
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

    from increment.estimation.engine import Method

    results = analysis.run(decision_method=Method(name="unadjusted"))
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

    rows = _legacy_fixed10_rows()
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
    rows = _legacy_fixed10_rows()
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
        "moments_format": 11,
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


def test_moments_from_summaries_replays_mean_and_conversion_summary(tmp_path):
    import pandas as pd

    from increment import Analysis, moments_from_summaries

    units = [
        {"unit": f"{group}-{index}", "arm": group, "value": value, "converted": converted}
        for group, values, conversions in (
            ("control", [0.0, 1.0, -1.0], [0, 1, 0]),
            ("treatment", [2.0, 3.0, 1.0], [1, 1, 0]),
        )
        for index, (value, converted) in enumerate(zip(values, conversions, strict=True))
    ]
    offset_units = [{**unit, "value": 1_000_000_000.0 + float(unit["value"])} for unit in units]
    original = Analysis.from_unit_summary(
        pd.DataFrame(offset_units),
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean", "converted": "conversion"},
        experiment_id="summary-audit",
    )

    import pyarrow.parquet as pq

    source_export = tmp_path / "source.parquet"
    original.export(source_export)
    source_rows = {
        (row["metric"], row["group_id"]): row for row in pq.read_table(source_export).to_pylist()
    }
    summaries = [
        {"metric": "value", "group_id": group, "n": 3, "mean": mean, "var": 1.0}
        for group, mean in (("control", 1_000_000_000.0), ("treatment", 1_000_000_002.0))
    ] + [
        {"metric": "converted", "group_id": group, "n": 3, "successes": successes}
        for group, successes in (("control", 1), ("treatment", 2))
    ]
    exported = moments_from_summaries(
        summaries,
        metrics={"value": "mean", "converted": "conversion"},
        control="control",
        study_id="summary-audit",
    )
    replay = Analysis.from_moments(
        exported,
        metrics={"value": "mean", "converted": "conversion"},
        control="control",
    )
    original_rows = {row.metric: row for row in _lift_rows(original.run())}
    replay_rows = {row.metric: row for row in _lift_rows(replay.run())}
    for metric, original_row in original_rows.items():
        replay_row = replay_rows[metric]
        assert replay_row.require_lift().value == pytest.approx(
            original_row.require_lift().value, abs=1e-10
        )
        assert replay_row.require_lift().lb == pytest.approx(
            original_row.require_lift().lb, abs=1e-10
        )
        assert replay_row.require_lift().ub == pytest.approx(
            original_row.require_lift().ub, abs=1e-10
        )
    assert {row["moments_format"] for row in exported} == {12}
    assert all(row["decision_plan"] and row["source_identity"] for row in exported)
    from increment.sources import WINSORIZATION_MOMENT_FIELDS

    assert all(all(field in row for field in WINSORIZATION_MOMENT_FIELDS) for row in exported)

    for row in exported:
        source_row = source_rows[(row["metric"], row["group_id"])]
        assert row["decision_plan"] == source_row["decision_plan"]
        assert row["source_identity"] == source_row["source_identity"]


def test_moments_from_summaries_refuses_n_one_mean_summary():
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [
                {"metric": "value", "group_id": "control", "n": 1, "mean": 2.0, "var": 9.0},
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 3.0, "var": 7.0},
            ],
            metrics={"value": "mean"},
            control="control",
        )
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_preserves_optional_assignment_counts():
    import json

    from increment import Analysis, moments_from_summaries
    from increment.sources import ASSIGNMENT_COUNTS_FIELD

    rows = moments_from_summaries(
        [
            {"metric": "value", "group_id": "control", "n": 2, "mean": 1.0, "var": 0.5},
            {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 0.5},
        ],
        metrics={"value": "mean"},
        control="control",
        assignment_counts={"control": 5, "treatment": 7},
    )
    for row in rows:
        payload = row[ASSIGNMENT_COUNTS_FIELD]
        assert isinstance(payload, str)
        assert json.loads(payload) == {"control": 5, "treatment": 7}
    analysis = Analysis.from_moments(rows, metrics={"value": "mean"}, control="control")
    assert analysis.unit_counts() == {"control": 5, "treatment": 7}
    from increment.estimation.diagnostics import SRMResult

    result = analysis.srm(
        expected={"control": 5 / 12, "treatment": 7 / 12},
        inference="fixed",
    )
    assert isinstance(result, SRMResult)
    assert result.observed == {"control": 5, "treatment": 7}


@pytest.mark.parametrize(
    "assignment_counts",
    [
        {"control": -1, "treatment": 1},
        {"control": True, "treatment": 1},
        {"control": 1},
    ],
)
def test_moments_from_summaries_rejects_invalid_assignment_counts(assignment_counts):
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [
                {"metric": "value", "group_id": "control", "n": 2, "mean": 1.0, "var": 0.5},
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 0.5},
            ],
            metrics={"value": "mean"},
            control="control",
            assignment_counts=assignment_counts,
        )
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_without_assignment_counts_refuses_count_operations():
    from increment import Analysis, moments_from_summaries
    from increment.errors import CodedError
    from increment.semantics.design import Randomized

    rows = moments_from_summaries(
        [
            {"metric": "value", "group_id": "control", "n": 2, "mean": 1.0, "var": 0.5},
            {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 0.5},
        ],
        metrics={"value": "mean"},
        control="control",
    )
    analysis = Analysis.from_moments(
        rows,
        metrics={"value": "mean"},
        design=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
            allocation_scheme="independent",
        ),
    )
    with pytest.raises(CodedError) as counts:
        analysis.unit_counts()
    assert counts.value.code == "source.moments.assignment_counts"
    with pytest.raises(CodedError) as srm:
        analysis.srm(expected={"control": 0.5, "treatment": 0.5})
    assert srm.value.code == "source.moments.assignment_counts"
    allocation = next(
        check
        for check in analysis.health_check(
            expected_allocation={"control": 0.5, "treatment": 0.5}
        ).checks
        if check.check == "assigned_allocation"
    )
    assert allocation.availability == "unavailable"
    assert allocation.code == "source.moments.assignment_counts"


@pytest.mark.parametrize(
    "row, metric_type",
    [
        (
            {"metric": "value", "group_id": "control", "n": 2, "successes": 1, "mean": 0.9},
            "conversion",
        ),
        (
            {"metric": "value", "group_id": "control", "n": 2, "successes": 1, "cluster": "store"},
            "conversion",
        ),
    ],
)
def test_moments_from_summaries_rejects_unexpected_row_fields(row, metric_type):
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [row],
            metrics={"value": metric_type},
            control="control",
        )
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_refuses_metrics_without_control_or_rows():
    from increment import moments_from_summaries
    from increment.errors import CodedError

    for rows in (
        [{"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 0.5}],
        [{"metric": "other", "group_id": "control", "n": 2, "mean": 1.0, "var": 0.5}],
    ):
        with pytest.raises(CodedError) as raised:
            moments_from_summaries(
                rows,
                metrics={"value": "mean"},
                control="control",
            )
        assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_conversion_family_matches_canonicalizer():
    from increment import moments_from_summaries
    from increment._bernoulli_moments import canonical_bernoulli_family

    summarized = moments_from_summaries(
        [
            {"metric": "converted", "group_id": "control", "n": 7, "successes": 5},
            {"metric": "converted", "group_id": "treatment", "n": 7, "successes": 5},
        ],
        metrics={"converted": "conversion"},
        control="control",
    )
    expected = canonical_bernoulli_family(7, 5)
    for row in summarized:
        assert tuple(row[key] for key in ("ref_y", "cy1", "cy2")) == expected


def test_moments_from_summaries_refuses_nonfinite_derived_mean_moments():
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [
                {"metric": "value", "group_id": "control", "n": 100, "mean": 0.0, "var": 1e307},
                {"metric": "value", "group_id": "treatment", "n": 100, "mean": 1.0, "var": 1.0},
            ],
            metrics={"value": "mean"},
            control="control",
        )
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_rejects_invalid_study_id_before_plan_compile(monkeypatch):
    import increment.summaries as summaries
    from increment import moments_from_summaries
    from increment.errors import CodedError

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("invalid study ID must be rejected before plan compilation")

    monkeypatch.setattr(summaries, "_wire_metadata", unexpected_compile)
    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [
                {"metric": "value", "group_id": "control", "n": 2, "mean": 1.0, "var": 0.5},
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 0.5},
            ],
            metrics={"value": "mean"},
            control="control",
            study_id="",
        )
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_rejects_unknown_metric_type():
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries([], metrics={"value": "maen"}, control="control")
    assert raised.value.code == "moments.summaries.invalid"


def test_moments_from_summaries_runs_two_arm_example():
    from increment import Analysis, moments_from_summaries

    rows = moments_from_summaries(
        [
            {"metric": "value", "group_id": "control", "n": 100, "mean": 100.0, "var": 4.0},
            {"metric": "value", "group_id": "treatment", "n": 100, "mean": 110.0, "var": 4.0},
        ],
        metrics={"value": "mean"},
        control="control",
    )
    assert Analysis.from_moments(rows, metrics={"value": "mean"}, control="control").run()


@pytest.mark.parametrize(
    "summary_rows, metric_types",
    [
        (
            [
                {"metric": "value", "group_id": "control", "n": 2, "mean": 1.0, "var": -1.0},
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 1.0},
            ],
            {"value": "mean"},
        ),
        (
            [
                {
                    "metric": "value",
                    "group_id": "control",
                    "n": 2,
                    "mean": float("nan"),
                    "var": 1.0,
                },
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 1.0},
            ],
            {"value": "mean"},
        ),
        (
            [
                {
                    "metric": "value",
                    "group_id": "control",
                    "n": 2,
                    "mean": 1.0,
                    "var": float("inf"),
                },
                {"metric": "value", "group_id": "treatment", "n": 2, "mean": 2.0, "var": 1.0},
            ],
            {"value": "mean"},
        ),
        (
            [
                {"metric": "converted", "group_id": "control", "n": 2, "successes": 3},
                {"metric": "converted", "group_id": "treatment", "n": 2, "successes": 1},
            ],
            {"converted": "conversion"},
        ),
        (
            [
                {"metric": "converted", "group_id": "control", "n": 2, "successes": 1.5},
                {"metric": "converted", "group_id": "treatment", "n": 2, "successes": 1},
            ],
            {"converted": "conversion"},
        ),
    ],
)
def test_moments_from_summaries_refuses_invalid_summary_values(summary_rows, metric_types):
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(summary_rows, metrics=metric_types, control="control")
    assert raised.value.code == "moments.summaries.invalid"


@pytest.mark.parametrize("metric_type", ["ratio", "covariate", "cluster", "sequential"])
def test_moments_from_summaries_refuses_unsupported_requests(metric_type):
    from increment import moments_from_summaries
    from increment.errors import CodedError

    with pytest.raises(CodedError) as raised:
        moments_from_summaries(
            [],
            metrics={"value": metric_type},
            control="control",
        )
    assert raised.value.code == "moments.summaries.unsupported"
    assert raised.value.context["route"] == "from_unit_summary"


def test_boolean_moment_control_preserves_count_only_literal_false(tmp_path):
    import json

    import pandas as pd
    import pyarrow.parquet as pq

    original = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"y": "mean"},
    )
    path = tmp_path / "literal-false-count-only.parquet"
    original.export(path)
    rows = pq.read_table(path).to_pylist()
    counts = {"True": 2, "treatment": 2, "False": 2}
    expected_counts = {"true": 2, "treatment": 2, "False": 2}
    for row in rows:
        row["assignment_counts"] = json.dumps(counts)
        row["group_id"] = True if row["group_id"] == "treatment" else "treatment"
    replay = Analysis.from_moments(rows, metrics={"y": "mean"}, control="False")
    assert replay.unit_counts() == expected_counts
