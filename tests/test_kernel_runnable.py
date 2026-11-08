from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from increment import AnalysisPlan, InferenceSpec, Randomized, run
from increment.builders import from_frame, from_moments
from increment.errors import CodedError
from increment.evidence import EvidenceCapabilities
from increment.kernel.contracts import Registration, View
from increment.kernel.experiment import Experiment, SourceReference
from increment.kernel.resolver import resolve, runnable
from increment.results import BreakoutEstimates
from increment.workflows import export_moments


def test_resolve_refuses_scope_grain_missing_from_declared_capabilities():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(name="events", uri="frame://events", capabilities=capabilities),
        ),
        views=(View(view="daily", grain="daily"),),
    )

    refusals = resolve("randomized.unadjusted", experiment, "orders")

    assert tuple(refusal.code for refusal in refusals) == ("evidence.grain_unavailable",)

    assert tuple(
        refusal.code
        for refusal in runnable(experiment, "orders")["randomized.unadjusted"].static_refusals
    ) == ("evidence.grain_unavailable",)


def test_runnable_uses_recorded_capabilities_after_json_restore():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total", "daily"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    supported = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
                capabilities=capabilities,
            ),
        ),
        views=(View(view="daily", grain="daily"),),
    )
    unsupported = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
                capabilities=replace(capabilities, grains=frozenset({"total"})),
            ),
        ),
        views=(View(view="daily", grain="daily"),),
    )

    supported_loaded = Experiment.from_json(supported.to_json())
    unsupported_loaded = Experiment.from_json(unsupported.to_json())

    assert supported_loaded.source_refs[0].schema == unsupported_loaded.source_refs[0].schema
    assert runnable(supported_loaded, "orders")["randomized.unadjusted"].static_refusals == ()
    assert tuple(
        refusal.code
        for refusal in runnable(unsupported_loaded, "orders")[
            "randomized.unadjusted"
        ].static_refusals
    ) == ("evidence.grain_unavailable",)


def test_runnable_reports_unknown_without_recorded_capabilities_after_json_restore():
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(
                name="events",
                uri="frame://events",
                schema={"orders": "float64"},
            ),
        ),
        views=(
            View(
                view="daily",
                grain="daily",
                segment=({"property": "country", "source": None},),
            ),
        ),
    )

    restored = Experiment.from_json(experiment.to_json())

    assert runnable(restored, "orders")["randomized.unadjusted"].unknown == ("events.capabilities",)
    assert runnable(restored, "orders")["randomized.unadjusted"].static_refusals == ()


def test_runnable_names_authored_source_and_selection_for_unavailable_breakouts():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=("country",),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=True,
        raw_outcomes=True,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )
    experiment = Experiment(
        experiment_id="checkout",
        design={"mechanism": "randomized"},
        metrics={"orders": {"kind": "mean"}},
        registration=Registration(registration_hash="registration-hash"),
        source_refs=(
            SourceReference(name="events", uri="frame://events", capabilities=capabilities),
        ),
        views=(
            View(
                view="breakout",
                grain="total",
                segment=({"property": "unknown", "source": None},),
            ),
        ),
    )

    for candidate in (experiment, Experiment.from_json(experiment.to_json())):
        statuses = runnable(candidate, "orders")
        assert [
            (refusal.code, dict(refusal.context))
            for refusal in statuses["randomized.unadjusted"].static_refusals
        ] == [
            (
                "evidence.breakout_unavailable",
                {
                    "constructor": "events",
                    "requested": ("unknown",),
                    "offered": ("country",),
                    "route": "use a source with the requested breakout dimensions",
                },
            )
        ]
        assert {
            refusal.context["constructor"]
            for status in statuses.values()
            for refusal in status.static_refusals
            if "constructor" in refusal.context
        } == {"events"}


def test_multicovariate_runnability_uses_alternative_evidence_routes():
    capabilities = EvidenceCapabilities(
        kind="unit_summary",
        grains=frozenset({"total"}),
        breakouts=(),
        shape="unit_summary",
        operations=frozenset(),
        unit_rows=False,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset({"unit"}),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=frozenset(),
    )

    def experiment_for(declared):
        return Experiment(
            experiment_id="checkout",
            design={"mechanism": "randomized"},
            metrics={"orders": {"kind": "mean"}},
            registration=Registration(registration_hash="registration-hash"),
            source_refs=(
                SourceReference(
                    name="events",
                    uri="frame://events",
                    capabilities=declared,
                ),
            ),
            views=(View(view="run", grain="total"),),
        )

    refused = runnable(experiment_for(capabilities), "orders")[
        "randomized.cuped_multicovariate"
    ].static_refusals
    alternative = runnable(experiment_for(replace(capabilities, unit_rows=True)), "orders")[
        "randomized.cuped_multicovariate"
    ].static_refusals

    assert tuple(refusal.code for refusal in refused) == (
        "evidence.operation_unavailable",
        "evidence.unit_rows_unavailable",
    )
    assert alternative == ()


def _frame_analysis(shape):
    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(40)],
            "arm": ["control", "treatment"] * 20,
            "value": [float(i % 7) for i in range(40)],
            "day": "2025-01-01",
        }
    )
    if shape == "summary":
        return from_frame(
            frame,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
        )
    return from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
        observation_end="2025-01-02",
    )


def _moments_recording_a_trigger(tmp_path):
    """Fixed-horizon moments rows that record a trigger but carry no triggered population."""
    import pyarrow.parquet as pq

    path = tmp_path / "moments.parquet"
    summary, summary_reader = _frame_analysis("summary")
    export_moments(summary, summary_reader, path)
    rows = [{**row, "trigger_name": "fired"} for row in pq.read_table(path).to_pylist()]
    return from_moments(rows, metrics={"value": "mean"}, control="control")


def _preflight_and_run(experiment, reader):
    refusals = runnable(experiment, "value")["randomized.unadjusted"].static_refusals
    with pytest.raises(CodedError) as executed:
        run(experiment, reader)
    return [(refusal.code, dict(refusal.context)) for refusal in refusals], executed.value


@pytest.mark.parametrize("source", ["summary", "panel", "moments_with_trigger"])
def test_triggered_view_without_triggered_evidence_refuses_identically(source, tmp_path):
    experiment, reader = (
        _moments_recording_a_trigger(tmp_path)
        if source == "moments_with_trigger"
        else _frame_analysis(source)
    )
    preflight, executed = _preflight_and_run(
        experiment.with_views(View.overall(population="triggered")), reader
    )

    assert executed.code == "facade.analysis.trigger_unsupported"
    assert preflight == [(executed.code, dict(executed.context))]
    assert executed.context["experiment"] == experiment.experiment_id


@pytest.mark.parametrize("shape", ["summary", "panel"])
def test_breakout_view_without_declared_breakouts_refuses_identically(shape):
    experiment, reader = _frame_analysis(shape)
    preflight, executed = _preflight_and_run(experiment.with_views(View(view="breakout")), reader)

    assert executed.code == "evidence.breakout_unavailable"
    assert preflight == [(executed.code, dict(executed.context))]
    assert executed.context["requested"] == ()
    assert executed.context["constructor"] == "from_frame"


def test_runnable_accepts_all_breakout_for_registered_sequential_dimension():
    frame = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(96)],
            "arm": ["control", "treatment"] * 48,
            "outcome": [0, 1, 1, 1] * 24,
            "segment": ["x", "x", "y", "y"] * 24,
            "exposure": "2025-01-01",
        }
    )
    experiment, reader = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="arm",
        design=Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5}),
        metrics={"outcome": "conversion"},
        plan=AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", segments={"segment": ("x", "y")}),
        ),
        exposure_date="exposure",
    )
    every_breakout = replace(experiment, views=(View(view="breakout", grain="total"),))

    refusals = runnable(every_breakout, "outcome")["randomized.unadjusted"].static_refusals

    assert [(refusal.code, dict(refusal.context)) for refusal in refusals] == []
    rows = run(every_breakout, reader, metrics=["outcome"]).readout()
    assert isinstance(rows, BreakoutEstimates)
    assert {(row.dimension, row.dimension_value) for row in rows} == {
        ("segment", "x"),
        ("segment", "y"),
    }
