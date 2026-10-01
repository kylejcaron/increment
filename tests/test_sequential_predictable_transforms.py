"""Transforms admitted under sequential inference iff predictable at each reveal."""

from fractions import Fraction as F

import numpy as np
import pandas as pd
import polars as pl
import pytest

from increment import Analysis, Method, PredeclaredAdjustment, fit_predeclared_adjustment
from increment.errors import CodedError
from increment.estimation.arm_contract import (
    AnalysisAxes,
    ArmCompatibilityRequest,
    FamilyPolicy,
    MethodCapability,
    MetricCapabilities,
    RelativeDecisionPolicy,
    Supported,
    Unsupported,
    arm_runtime_support,
)
from increment.estimation.sequential import AlwaysValid, AsymptoticMean
from increment.frame import MetricSpec
from increment.semantics.assignment import ParallelAssignment
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, InferenceSpec
from increment.sequential_state import validate_sequential_transform
from tests.asymptotic_cases import mean_records, mean_registration
from tests.sequential_cases import registration

_DESIGN = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})


def _frame(control, treatment):
    return pd.DataFrame(
        [
            {"unit": r["unit_id"], "arm": r["group_id"], "exposure": i, **r["values"]}
            for i, r in enumerate(mean_records(control, treatment))
        ]
    )


def _run(frame, spec, plan):
    frame = frame.copy()
    if "exposure" not in frame:
        frame["exposure"] = range(len(frame))
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[spec],
        experiment_id="experiment",
        plan=plan,
        design=_DESIGN,
        exposure_date="exposure",
    )
    return analysis, analysis.run()[0]


_ASYMPTOTIC = AnalysisPlan(primary="outcome", inference=InferenceSpec(kind="asymptotic_mean"))


def test_fixed_winsorization_runs_and_retains_clipped_scalars():
    spec = MetricSpec(name="outcome", type="mean", winsorization={"upper_value": 20.0})
    analysis, row = _run(_frame([1, 2, 3, 40] * 30, [7, 10, 13, 50] * 30), spec, _ASYMPTOTIC)
    assert row.inference == "asymptotic_mean" and row.lift is not None
    means = {s.group_id: s.mean[0] for s in analysis.sequential_snapshot().states}
    assert means == {"control": F(13, 2), "treatment": F(25, 2)}


def test_percentile_winsorization_refuses_as_unpredictable():
    spec = MetricSpec(name="outcome", type="mean", winsorization={"upper_percentile": 0.9})
    with pytest.raises(CodedError) as raised:
        _run(_frame([1, 2, 3, 40] * 30, [7, 10, 13, 50] * 30), spec, _ASYMPTOTIC)
    assert raised.value.code == "sequential.transform.unpredictable"


@pytest.mark.parametrize(
    "bounds", [{"upper_value": 20.0}, {"lower_value": 2.0, "upper_value": 20.0}]
)
def test_fixed_threshold_passes_the_shared_gate_every_route_consults(bounds):
    validate_sequential_transform(MetricSpec(name="outcome", type="mean", winsorization=bounds))


def test_percentile_threshold_refused_at_the_shared_gate():
    spec = MetricSpec(
        name="outcome", type="mean", winsorization={"lower_value": 1.0, "upper_percentile": 0.99}
    )
    with pytest.raises(CodedError) as raised:
        validate_sequential_transform(spec)
    assert raised.value.code == "sequential.transform.unpredictable"


def _pre_period_frame(rng, n=400):
    arms = np.array(["control", "treatment"] * (n // 2))
    x = rng.normal(10.0, 3.0, size=n)
    return pd.DataFrame(
        {
            "unit": [f"p{i:05d}" for i in range(n)],
            "arm": arms,
            "outcome_pre": 5.0 + 0.9 * x + rng.normal(0.0, 1.0, size=n),
            "covariate_pre": x,
        }
    )


def _experiment_frame(rng, n=400):
    arms = np.array(["control", "treatment"] * (n // 2))
    x = rng.normal(10.0, 3.0, size=n)
    y = 5.0 + 0.9 * x + rng.normal(0.0, 1.0, size=n) + (arms == "treatment")
    return pd.DataFrame(
        {"unit": [f"u{i:05d}" for i in range(n)], "arm": arms, "outcome": y, "covariate": x}
    )


_CUPED = Method(name="cuped", variance_reduction="cuped")


def test_predeclared_cuped_narrows_the_asymptotic_interval_and_labels_the_row():
    rng = np.random.default_rng(7)
    adjustment = fit_predeclared_adjustment(
        _pre_period_frame(rng),
        unit="unit",
        group="arm",
        control="control",
        outcome="outcome_pre",
        covariate="covariate_pre",
    )
    frame = _experiment_frame(rng)
    _, plain = _run(
        frame, MetricSpec(name="outcome", type="mean", covariate="covariate"), _ASYMPTOTIC
    )
    adjusted_plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", adjustments={"outcome": adjustment}),
    )
    spec = MetricSpec(name="outcome", type="mean", covariate="covariate", decision_method=_CUPED)
    analysis, adjusted = _run(frame, spec, adjusted_plan)
    assert plain.method == "unadjusted" and adjusted.method == "cuped"
    assert adjusted.lift.ub - adjusted.lift.lb < plain.lift.ub - plain.lift.lb
    model = analysis.sequential_snapshot().registration.models[0]
    assert model.adjustment == adjustment

    # The coefficient never reads an in-experiment outcome: rewriting every
    # outcome moves the retained state but leaves the registration untouched.
    mutated = frame.assign(outcome=frame["outcome"] * 3 + 100)
    other, _ = _run(mutated, spec, adjusted_plan)
    before, after = analysis.sequential_snapshot(), other.sequential_snapshot()
    assert before.registration_id == after.registration_id
    assert before.prefix_id != after.prefix_id


def test_in_experiment_coefficient_is_retained_as_joint_moments_on_the_asymptotic_route():
    rng = np.random.default_rng(7)
    spec = MetricSpec(name="outcome", type="mean", covariate="covariate", decision_method=_CUPED)
    analysis, row = _run(_experiment_frame(rng), spec, _ASYMPTOTIC)
    model = analysis.sequential_snapshot().registration.models[0]
    assert model.law == "adjusted_mean" and model.adjustment is None
    assert row.method == "cuped"
    assert {len(s.mean) for s in analysis.sequential_snapshot().states} == {2}


def test_in_experiment_coefficient_still_refuses_on_the_exact_routes():
    from increment.sequential_state import validate_sequential_methods

    with pytest.raises(CodedError) as raised:
        validate_sequential_methods(registration("bernoulli"), "outcome", (_CUPED,))
    assert raised.value.code == "sequential.transform.unpredictable"


def test_capture_retains_the_adjusted_scalar_exactly():
    adjustment = PredeclaredAdjustment(coefficient=F(1, 2), center=F(4))
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", adjustments={"outcome": adjustment}),
    )
    frame = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": ["control", "control", "treatment", "treatment"],
            "outcome": [10.0, 12.0, 20.0, 24.0],
            "covariate": [2.0, 6.0, 4.0, 8.0],
        }
    )
    spec = MetricSpec(name="outcome", type="mean", covariate="covariate", decision_method=_CUPED)
    analysis, _ = _run(frame, spec, plan)
    means = {s.group_id: s.mean[0] for s in analysis.sequential_snapshot().states}
    # control: 10 - (2-4)/2 = 11, 12 - (6-4)/2 = 11; treatment: 20, 22.
    assert means == {"control": F(11), "treatment": F(21)}


def _request(inference, variance_reduction):
    return ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification="randomized",
            view="total",
            segmented=False,
            completed_windows_only=False,
            population="assigned",
            variance_adjustment="none",
        ),
        dependence="iid",
        inference=inference,
        estimand="itt",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative="two-sided",
            null_lift=0.0,
            signed_ratio=isinstance(inference, AsymptoticMean),
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ),
        methods=(
            MethodCapability(
                role="decision", estimator="cuped", variance_reduction=variance_reduction
            ),
        ),
        prior_present=False,
    )


@pytest.mark.parametrize(
    "inference",
    [
        AsymptoticMean(registration=mean_registration()),
        AlwaysValid(registration=registration("bernoulli")),
    ],
    ids=["asymptotic_mean", "always_valid"],
)
def test_arm_contract_admits_retained_moments_only_on_the_asymptotic_route(inference):
    predeclared = arm_runtime_support(_request(inference, "predeclared_cuped"))
    if isinstance(inference, AsymptoticMean):
        assert isinstance(predeclared, Supported)
        assert predeclared.reference == "sequential_boundary"
    else:
        assert isinstance(predeclared, Unsupported)
        assert predeclared.refusal_code == "arm.adjustment.sequential_cuped"
    fitted = arm_runtime_support(_request(inference, "cuped"))
    assert isinstance(fitted, Unsupported)
    assert fitted.refusal_code == "arm.adjustment.sequential_cuped"
    retained = arm_runtime_support(_request(inference, "retained_cuped"))
    if isinstance(inference, AsymptoticMean):
        assert isinstance(retained, Supported)
    else:
        assert isinstance(retained, Unsupported)
        assert retained.refusal_code == "arm.adjustment.sequential_cuped"


def test_panel_continuation_uses_natural_exposure_order_and_rejects_rewrites():
    from increment.frame import from_unit_panel
    from increment.sequential_state import snapshot_from_json

    rows = pl.DataFrame(
        {
            "u": ["c", "t"],
            "g": ["C", "T"],
            "day": ["d2", "d10"],
            "exposure": ["d2", "d10"],
            "y": [1.0, 2.0],
        }
    )
    plan = AnalysisPlan(primary="y", inference=InferenceSpec(kind="asymptotic_mean"))

    def source(frame):
        return from_unit_panel(
            frame,
            unit="u",
            group="g",
            date="day",
            exposure_date="exposure",
            control="C",
            metrics=[MetricSpec(name="y", window_days=1)],
            design=Randomized(control_group="C", allocation={"C": 0.5, "T": 0.5}),
            plan=plan,
        )

    first = source(rows).capture_sequential(as_of="d2", finalized=True)
    replayed = snapshot_from_json(first.model_dump_json())
    continued = source(rows).capture_sequential(as_of="d10", finalized=True, previous=replayed)
    fresh = source(rows).capture_sequential(as_of="d10", finalized=True)
    assert continued.model_dump(exclude={"parent_id", "ancestors"}) == fresh.model_dump(
        exclude={"parent_id", "ancestors"}
    )
    assert continued.parent_id == replayed.prefix_id
    assert snapshot_from_json(continued.model_dump_json()) == continued

    rewritten = rows.with_columns(
        pl.when(pl.col("u") == "c").then(99.0).otherwise(pl.col("y")).alias("y")
    )
    with pytest.raises(CodedError) as raised:
        source(rewritten).capture_sequential(as_of="d10", finalized=True, previous=replayed)
    assert raised.value.code == "sequential.continuation.rewrite"


@pytest.mark.parametrize(
    "aliases",
    [
        ("__day_idx__", "__exposure__"),
        ("__observable_days__", "__exposure___right"),
        ("___observable_days__", "__observable_days__"),
    ],
    ids=["day-anchor", "maturity-numerator", "maturity-denominator"],
)
@pytest.mark.parametrize("backend", ["polars", "arrow"])
def test_panel_sequential_capture_preserves_metric_columns_named_like_scratch(backend, aliases):
    from increment.frame import from_unit_panel

    rows = pl.DataFrame(
        [
            {
                "u": str(unit),
                "g": "C" if unit < 2 else "T",
                "day": f"d{day}",
                "exposure": "d2",
                "y": 10000.0 if day == 1 else float(2 * day - 2 + unit) / 16,
                "den": 10000.0 if day == 1 else float(day - 1 + unit) / 16,
            }
            for unit in range(4)
            for day in (1, 2, 3)
        ]
    )
    plan = AnalysisPlan(
        primary="mean", secondaries=["ratio"], inference=InferenceSpec(kind="asymptotic_mean")
    )

    def capture(alias):
        value, denominator = aliases if alias else ("y", "den")
        frame = rows.rename({"y": value, "den": denominator}) if alias else rows
        return from_unit_panel(
            frame.to_arrow() if backend == "arrow" else frame,
            unit="u",
            group="g",
            date="day",
            exposure_date="exposure",
            control="C",
            metrics=[
                MetricSpec(name="mean", value_column=value, window_days=2),
                MetricSpec(
                    name="ratio",
                    type="ratio",
                    numerator=value,
                    denominator=denominator,
                    window_days=2,
                ),
            ],
            design=Randomized(control_group="C", allocation={"C": 0.5, "T": 0.5}),
            plan=plan,
        ).capture_sequential(as_of="d4", finalized=True)

    ordinary, aliased = capture(False), capture(True)
    assert [(record.unit_id, record.group_id) for record in aliased.records] == [
        ("0", "C"),
        ("1", "C"),
        ("2", "T"),
        ("3", "T"),
    ]
    assert aliased.states == ordinary.states
    assert {(state.group_id, state.mean) for state in aliased.states} == {
        ("C", (F(7, 16),)),
        ("T", (F(11, 16),)),
        ("C", (F(7, 16), F(1, 4))),
        ("T", (F(11, 16), F(1, 2))),
    }


def _missing_zero_summary_snapshot(*, outcome, denominator=1.0, ratio=False):
    frame = pl.DataFrame(
        {
            "unit": ["c", "t"],
            "arm": ["control", "treatment"],
            "exposure": [0, 1],
            "outcome": [4.0, outcome],
            "denominator": [2.0, denominator],
        }
    )
    spec = (
        MetricSpec(
            name="outcome",
            type="ratio",
            numerator="outcome",
            denominator="denominator",
            missing="zero",
        )
        if ratio
        else MetricSpec(name="outcome", missing="zero")
    )
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        metrics=[spec],
        experiment_id="missing-zero",
        plan=_ASYMPTOTIC,
        design=_DESIGN,
        exposure_date="exposure",
    )
    return analysis.sequential_snapshot()


def test_sequential_missing_zero_normalizes_nan_and_null_identically():
    from increment.sequential_state import snapshot_from_json

    scalar = [_missing_zero_summary_snapshot(outcome=value) for value in (float("nan"), None, 0.0)]
    expected_scalar = scalar[-1]
    assert {
        state.group_id: (state.n, state.mean, state.scatter) for state in expected_scalar.states
    } == {
        "control": (1, (F(4),), ((F(0),),)),
        "treatment": (1, (F(0),), ((F(0),),)),
    }
    for snapshot in scalar:
        assert snapshot.records == expected_scalar.records
        assert snapshot.states == expected_scalar.states
        assert snapshot.prefix_id == expected_scalar.prefix_id
        assert snapshot_from_json(snapshot.model_dump_json()) == snapshot

    for coordinate in ("outcome", "denominator"):
        snapshots = []
        for value in (float("nan"), None, 0.0):
            snapshots.append(
                _missing_zero_summary_snapshot(
                    outcome=value if coordinate == "outcome" else 6.0,
                    denominator=value if coordinate == "denominator" else 3.0,
                    ratio=True,
                )
            )
        expected = snapshots[-1]
        expected_treatment = (F(0), F(3)) if coordinate == "outcome" else (F(6), F(0))
        states = {state.group_id: state for state in expected.states}
        assert states["treatment"].mean == expected_treatment
        for snapshot in snapshots:
            assert snapshot.records == expected.records
            assert snapshot.states == expected.states
            assert snapshot.prefix_id == expected.prefix_id
            assert snapshot_from_json(snapshot.model_dump_json()) == snapshot


def test_sequential_missing_zero_keeps_covariate_and_refusal_policies_separate():
    adjustment = PredeclaredAdjustment(coefficient=F(1, 2), center=F(4))
    plan = AnalysisPlan(
        primary="outcome",
        inference=InferenceSpec(kind="asymptotic_mean", adjustments={"outcome": adjustment}),
    )
    frame = pl.DataFrame(
        {
            "unit": ["c", "t"],
            "arm": ["control", "treatment"],
            "exposure": [0, 1],
            "outcome": [4.0, float("nan")],
            "covariate": [4.0, None],
        }
    )

    def capture(covariate_missing):
        with pytest.warns(UserWarning):
            analysis = Analysis.from_unit_summary(
                frame,
                unit="unit",
                group="arm",
                metrics=[
                    MetricSpec(
                        name="outcome",
                        missing="zero",
                        covariate="covariate",
                        covariate_missing=covariate_missing,
                        decision_method=_CUPED,
                    )
                ],
                experiment_id=f"covariate-{covariate_missing}",
                plan=plan,
                design=_DESIGN,
                exposure_date="exposure",
            )
        return {state.group_id: state.mean[0] for state in analysis.sequential_snapshot().states}

    assert capture("impute") == {"control": F(4), "treatment": F(0)}
    assert capture("zero") == {"control": F(4), "treatment": F(2)}

    with pytest.raises(CodedError) as covariate_error:
        Analysis.from_unit_summary(
            frame,
            unit="unit",
            group="arm",
            metrics=[
                MetricSpec(
                    name="outcome",
                    missing="zero",
                    covariate="covariate",
                    covariate_missing="error",
                    decision_method=_CUPED,
                )
            ],
            experiment_id="covariate-error",
            plan=plan,
            design=_DESIGN,
            exposure_date="exposure",
        )
    assert covariate_error.value.code == "frame.validation.metric_covariate_column"

    for policy, code in (
        ("error", "frame.validation.metric_value_missing"),
        ("drop", "sequential.route.unsupported"),
    ):
        with pytest.raises(CodedError) as missing_error:
            Analysis.from_unit_summary(
                frame.drop("covariate"),
                unit="unit",
                group="arm",
                metrics=[MetricSpec(name="outcome", missing=policy)],
                experiment_id=f"outcome-{policy}",
                plan=_ASYMPTOTIC,
                design=_DESIGN,
                exposure_date="exposure",
            )
        assert missing_error.value.code == code

    infinite = frame.drop("covariate").with_columns(pl.lit(float("inf")).alias("outcome"))
    with pytest.raises(CodedError) as nonfinite:
        Analysis.from_unit_summary(
            infinite,
            unit="unit",
            group="arm",
            metrics=[
                MetricSpec(
                    name="outcome",
                    missing="zero",
                    winsorization={"upper_value": 10.0},
                )
            ],
            experiment_id="outcome-infinite",
            plan=_ASYMPTOTIC,
            design=_DESIGN,
            exposure_date="exposure",
        )
    assert nonfinite.value.code == "source.frame.non_finite"


def test_panel_missing_zero_checkpoint_reload_and_continuation_match_zero():
    from increment.frame import from_unit_panel
    from increment.sequential_state import snapshot_from_json

    def source(value):
        return from_unit_panel(
            pl.DataFrame(
                {
                    "u": ["c", "t"],
                    "g": ["C", "T"],
                    "day": ["d1", "d2"],
                    "exposure": ["d1", "d2"],
                    "y": [1.0, value],
                }
            ),
            unit="u",
            group="g",
            date="day",
            exposure_date="exposure",
            control="C",
            metrics=[MetricSpec(name="y", missing="zero", window_days=1)],
            design=Randomized(control_group="C", allocation={"C": 0.5, "T": 0.5}),
            plan=AnalysisPlan(primary="y", inference=InferenceSpec(kind="asymptotic_mean")),
            experiment_id="panel-missing-zero",
        )

    completed = []
    for value in (float("nan"), None, 0.0):
        first = source(value).capture_sequential(as_of="d1", finalized=True)
        restored = snapshot_from_json(first.model_dump_json())
        continued = source(value).capture_sequential(as_of="d2", finalized=True, previous=restored)
        fresh = source(value).capture_sequential(as_of="d2", finalized=True)
        assert continued.model_dump(exclude={"parent_id", "ancestors"}) == fresh.model_dump(
            exclude={"parent_id", "ancestors"}
        )
        assert continued.parent_id == restored.prefix_id
        assert snapshot_from_json(continued.model_dump_json()) == continued
        completed.append(continued)

    expected = completed[-1]
    for snapshot in completed:
        assert snapshot.records == expected.records
        assert snapshot.states == expected.states
        assert snapshot.prefix_id == expected.prefix_id
