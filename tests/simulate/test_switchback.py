from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.simulate.dgp import SwitchbackScenario, simulate_switchback_panel
from increment.simulate.runner import SwitchbackEvalResult, run_switchback_end_to_end


@pytest.mark.parametrize("entry", ["constructor", "model_validate", "model_validate_json"])
@pytest.mark.parametrize("field", ["residual_carryover", "unexpected_option"])
def test_switchback_scenario_rejects_extra_fields_coded(entry, field):
    import json

    from increment.errors import InvalidRequestError

    values: dict[str, Any] = {"n_units": 2, "n_cycles": 2, field: 1}
    with pytest.raises(InvalidRequestError) as exc_info:
        if entry == "constructor":
            SwitchbackScenario(**values)
        elif entry == "model_validate":
            SwitchbackScenario.model_validate(values)
        else:
            SwitchbackScenario.model_validate_json(json.dumps(values))
    error = exc_info.value
    assert error.code == "simulate.dgp.switchback.extra_fields"
    assert error.context["fields"] == (field,)
    with pytest.raises(TypeError):
        cast("dict[str, object]", error.context)["fields"] = ()
    with pytest.raises(TypeError):
        cast("list[str]", error.context["fields"])[0] = "changed"


def test_switchback_simulation_is_deterministic_and_has_temporal_dependence():
    scenario = SwitchbackScenario(
        n_units=12,
        n_cycles=3,
        treatment_effect=2.0,
        temporal_effect=0.5,
        seed=7,
    )
    first = simulate_switchback_panel(scenario)
    second = simulate_switchback_panel(scenario)
    control = simulate_switchback_panel(scenario.model_copy(update={"temporal_effect": 0.0}))
    assert first.equals(second)
    values = np.asarray(first.column("outcome"))
    control_values = np.asarray(control.column("outcome"))
    assert np.isfinite(values).all()
    assert not np.array_equal(values, control_values)
    assert np.mean(values - control_values) > 0.0


@pytest.mark.parametrize("law", ["normal", "centered_lognormal", "centered_gamma"])
def test_disabled_switchback_noise_preserves_legacy_seed_stream(law):
    """Freeze the old generator arithmetic and order-draw sequence independently."""
    import pyarrow as pa

    scenario = SwitchbackScenario(n_units=4, n_cycles=2, seed=7)
    rng = np.random.default_rng(7)
    intercepts = rng.normal(0, 1, 4)
    rows = []
    for u in range(4):
        previous = False
        for c in range(2):
            ct = rng.binomial(1, 0.5)
            groups = ("control", "treatment") if ct else ("treatment", "control")
            for period, group in enumerate(groups):
                for step in range(3):
                    residual = 0.5 if previous and step == 0 else 0.0
                    value = intercepts[u] + 0.25 * (c + 1) + 0.25 * period + 0.0 + residual
                    rows.append(
                        {
                            "unit": f"u{u:06d}",
                            "cycle": c,
                            "period": period,
                            "step": step,
                            "group": group,
                            "outcome": float(value),
                        }
                    )
                previous = group == "treatment"
    legacy = pa.Table.from_pylist(rows)
    declared = scenario.model_copy(
        update={
            "noise_distribution": law,
            "noise_sd": 0.0,
            "within_unit_correlation": 0.9,
        }
    )
    actual = simulate_switchback_panel(declared)
    assert actual.equals(legacy)
    assert actual.schema == legacy.schema


@pytest.mark.parametrize("law", ["normal", "centered_lognormal", "centered_gamma"])
def test_treatment_noise_survives_paired_contrasts_and_keeps_orders(law):
    scenario = SwitchbackScenario(
        n_units=4,
        n_cycles=3,
        unit_effect_sd=0,
        temporal_effect=0,
        washout_steps=0,
        observation_steps=1,
        seed=11,
        noise_distribution=law,
        noise_sd=2,
        within_unit_correlation=1,
    )
    noisy = simulate_switchback_panel(scenario)
    clean = simulate_switchback_panel(scenario.model_copy(update={"noise_sd": 0.0}))
    assert noisy.drop(["outcome"]).equals(clean.drop(["outcome"]))
    outcomes = np.asarray(noisy["outcome"]).reshape(4, 3, 2)
    groups = np.asarray(noisy["group"]).reshape(4, 3, 2)
    contrasts = np.where(groups[..., 1] == "treatment", outcomes[..., 1], outcomes[..., 0])
    assert np.all(contrasts == contrasts[:, :1])
    assert np.any(contrasts != 0)
    assert np.all(outcomes[groups == "control"] == 0)
    # Independent reconstruction of the first common innovation stream.
    rng = np.random.default_rng(np.random.SeedSequence([11, 1414]))
    if law == "normal":
        expected = rng.normal(size=4)
    elif law == "centered_lognormal":
        expected = (rng.lognormal(size=4) - np.exp(0.5)) / np.sqrt(np.e * np.expm1(1))
    else:
        expected = (rng.gamma(2, size=4) - 2) / np.sqrt(2)
    assert contrasts[:, 0] == pytest.approx(2 * expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("noise_sd", -1),
        ("noise_sd", float("nan")),
        ("within_unit_correlation", -0.1),
        ("within_unit_correlation", 1.1),
        ("period_effect_heterogeneity", float("inf")),
        ("treatment_period_effect", float("nan")),
        ("noise_distribution", "skew"),
    ],
)
def test_switchback_noise_rejects_invalid_declarations(field, value):
    with pytest.raises((ValidationError, ValueError)):
        SwitchbackScenario(n_units=4, n_cycles=1, **{field: value})


@pytest.mark.parametrize("law", ["normal", "centered_lognormal", "centered_gamma"])
def test_noisy_scenario_reaches_actual_estimation_and_pilot_sizing(law):
    from increment.estimation.contrast import estimate_contrast
    from increment.estimation.decision_types import ContrastDecisionProcedure
    from increment.frame import from_switchback_panel
    from increment.power import switchback_achieved_power, switchback_minimum_detectable_effect
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    approximation = UnitCycleTApproximation()
    scenario = SwitchbackScenario(
        n_units=4,
        n_cycles=5,
        observation_steps=1,
        washout_steps=0,
        probability_ct=0.9,
        treatment_effect=1,
        noise_distribution=law,
        noise_sd=1,
        within_unit_correlation=0.9,
        period_effect_heterogeneity=0.3,
        treatment_period_effect=0.2,
        seed=17,
    )
    src = from_switchback_panel(
        simulate_switchback_panel(scenario),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"outcome": "mean"},
        contrast_references={"outcome": approximation},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.9),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
    )
    proc = ContrastDecisionProcedure(
        metric="outcome",
        role="primary",
        alternative="two-sided",
        null_abs=0,
        alpha=0.05,
        reference=approximation,
    )
    result = estimate_contrast(src.contrast_stats(src.metrics[0]), proc).results[0]
    pilot = src.planning_baseline(src.metrics[0], delta_ref=1)
    prediction = switchback_achieved_power(4, 1, pilot, proc)
    assert prediction.standard_error == pytest.approx(result.standard_error)
    assert result.standard_error is not None
    assert result.standard_error > 0 and result.dof == 3
    assert result.estimate.lb is not None and result.estimate.ub is not None
    mde = switchback_minimum_detectable_effect(4, pilot, proc, target_power=0.8)
    if mde.mde_abs is not None:
        achieved = switchback_achieved_power(4, mde.mde_abs, pilot, proc)
        assert achieved.power == pytest.approx(0.8, abs=1e-8)
    else:
        assert mde.mde_unavailable_reason == "unattained"


def test_switchback_residual_carryover_is_temporal_and_washout_respected():
    base = SwitchbackScenario(
        n_units=20,
        n_cycles=2,
        treatment_effect=0.0,
        temporal_effect=0.0,
        washout_steps=1,
        observation_steps=2,
        true_carryover_order=0,
        carryover_amplitude=0.0,
        seed=9,
    )
    violation = simulate_switchback_panel(
        base.model_copy(update={"true_carryover_order": 2, "carryover_amplitude": 3.0})
    )
    clean = simulate_switchback_panel(base)
    observed = violation.to_pandas().set_index(["unit", "cycle", "period", "step"])
    reference = clean.to_pandas().set_index(["unit", "cycle", "period", "step"])
    delta = observed["outcome"] - reference["outcome"]
    # No future treatment can contaminate the first period, and washout rows
    # are excluded from the residual effect.
    assert (delta.loc[(slice(None), 0, 0, slice(None))] == 0.0).all()
    assert (delta.loc[(slice(None), slice(None), slice(None), 0)] == 0.0).all()
    assert (delta.loc[(slice(None), slice(None), slice(None), slice(1, 2))] >= 0.0).any()
    assert (delta.loc[(slice(None), 0, 1, slice(1, 2))] > 0.0).any()


def test_switchback_true_carryover_order_contaminates_only_the_declared_finite_history():
    """Acceptance: contamination applies only to the true finite-history
    steps, not every post-washout observation forever. With
    ``true_carryover_order=1`` and ``observation_steps=3``, only the first
    retained step after washout carries the residual; later steps are
    clean even though the prior period was treatment."""
    base = SwitchbackScenario(
        n_units=12,
        n_cycles=2,
        treatment_effect=0.0,
        temporal_effect=0.0,
        unit_effect_sd=0.0,
        washout_steps=1,
        observation_steps=3,
        true_carryover_order=1,
        carryover_amplitude=5.0,
        seed=3,
    )
    panel = simulate_switchback_panel(base).to_pandas()
    values = panel.set_index(["unit", "cycle", "period", "step"])["outcome"]

    observed = False
    for unit in panel["unit"].unique():
        for cycle in range(base.n_cycles):
            prior_group = panel.loc[
                (panel["unit"] == unit) & (panel["cycle"] == cycle) & (panel["period"] == 0),
                "group",
            ].iloc[0]
            if prior_group != "treatment":
                continue
            observed = True
            # Retained observation steps are washout_steps..washout_steps+observation_steps-1
            # == 1, 2, 3; only step 1 (washout_steps + true_carryover_order - 1) is
            # still contaminated, steps 2 and 3 are clean.
            contaminated = values.loc[(unit, cycle, 1, 1)]
            clean_1 = values.loc[(unit, cycle, 1, 2)]
            clean_2 = values.loc[(unit, cycle, 1, 3)]
            assert contaminated == pytest.approx(clean_1 + base.carryover_amplitude)
            assert clean_1 == pytest.approx(clean_2)
    assert observed


@pytest.mark.parametrize("true_order", [0, 1, 2])
@pytest.mark.parametrize("declared_order", [0, 1, 2])
@pytest.mark.parametrize("amplitude", [0.0, 3.0])
def test_switchback_evaluator_retains_declared_history(true_order, declared_order, amplitude):
    scenario = SwitchbackScenario(
        n_units=8,
        n_cycles=2,
        observation_steps=3,
        treatment_effect=6.0,
        unit_effect_sd=0.0,
        temporal_effect=0.0,
        true_carryover_order=true_order,
        declared_carryover_order=declared_order,
        carryover_amplitude=amplitude,
        seed=7,
    )
    result = run_switchback_end_to_end(scenario)
    contaminated = amplitude > 0.0 and true_order > declared_order
    assert result.assumption_violation is contaminated
    assert result.supported is (not contaminated)
    assert result.attempted == result.point_estimable == 1
    assert result.failed == result.excluded == 0
    bias = result.bias["outcome"]
    assert bias is not None
    if contaminated:
        assert bias < 0.0
        assert result.interval_estimable == 1
    else:
        # Each retained step contributes exactly 2. Equal realized unit
        # effects retain the point but cannot supply a variance estimate.
        assert bias == pytest.approx(0.0)
        assert result.interval_estimable == 0
        assert result.coverage_conditional["outcome"] is None
        assert result.coverage_unconditional["outcome"] == 0.0
        assert result.interval_unavailable_reasons == {"no_interval": 1}


@pytest.mark.parametrize("declared_order", [3, 4])
def test_switchback_scenario_refuses_empty_retained_window(declared_order):
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        SwitchbackScenario(
            n_units=2,
            n_cycles=1,
            observation_steps=3,
            declared_carryover_order=declared_order,
        )
    assert exc_info.value.code == "simulate.dgp.switchback.retained_window"
    assert exc_info.value.context["declared_carryover_order"] == declared_order
    assert exc_info.value.context["observation_steps"] == 3


@pytest.mark.parametrize(
    ("declared_order", "code"),
    [
        (-1, "model.field.range"),
        (True, "model.field.type"),
        (1.5, "model.field.type"),
        ("1", "model.field.type"),
    ],
)
def test_switchback_scenario_refuses_invalid_declared_order(declared_order, code):
    from increment.errors import InvalidRequestError

    error_type = InvalidRequestError if code is not None else ValidationError
    with pytest.raises(error_type) as raised:
        SwitchbackScenario(n_units=2, n_cycles=1, declared_carryover_order=declared_order)
    if code is not None:
        assert isinstance(raised.value, InvalidRequestError)
        assert raised.value.code == code


def test_switchback_evaluator_reports_effect_direction():
    scenario = SwitchbackScenario(n_units=20, n_cycles=3, treatment_effect=2.0, seed=7)
    result = run_switchback_end_to_end(scenario, replications=2)
    bias = result.bias["outcome"]
    assert bias is not None
    assert bias + scenario.treatment_effect > 0
    assert result.supported is True


@pytest.mark.parametrize("bad", [0, -5, 1.5, True, False])
def test_switchback_end_to_end_refuses_invalid_replications(bad):
    from increment.errors import InvalidRequestError

    scenario = SwitchbackScenario(n_units=20, n_cycles=3, treatment_effect=2.0, seed=7)
    with pytest.raises(InvalidRequestError) as exc_info:
        run_switchback_end_to_end(scenario, replications=bad)
    assert exc_info.value.code == "simulate.runner.replications_invalid"


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_supported_switchback_calibration_is_covered():
    scenario = SwitchbackScenario(
        n_units=120,
        n_cycles=4,
        treatment_effect=2.0,
        temporal_effect=0.5,
        seed=21,
    )
    result = run_switchback_end_to_end(scenario, replications=24)
    bias = result.bias["outcome"]
    coverage = result.coverage_unconditional["outcome"]
    assert bias is not None
    assert coverage is not None
    assert abs(bias) < 0.15
    assert 0.80 <= coverage <= 1.0
    assert result.attempted == 24
    assert result.failed == 0
    assert result.supported is True


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_switchback_assumption_violation_is_marked_and_not_supported_calibration():
    scenario = SwitchbackScenario(
        n_units=20,
        n_cycles=3,
        treatment_effect=2.0,
        true_carryover_order=2,
        carryover_amplitude=1.0,
        seed=7,
    )
    result = run_switchback_end_to_end(scenario, replications=16)
    assert result.assumption_violation is True
    assert result.supported is False
    coverage = result.coverage_conditional["outcome"]
    bias = result.bias["outcome"]
    assert coverage is not None
    assert bias is not None
    assert result.interval_estimable == result.attempted
    assert result.failed == 0
    assert coverage < 0.80
    assert abs(bias) > 0.25


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_switchback_null_effect_type_i_calibration():
    scenario = SwitchbackScenario(
        n_units=120, n_cycles=4, treatment_effect=0.0, temporal_effect=0.5, seed=31
    )
    result = run_switchback_end_to_end(scenario, replications=24)
    bias = result.bias["outcome"]
    coverage = result.coverage_unconditional["outcome"]
    assert bias is not None
    assert coverage is not None
    assert abs(bias) < 0.15
    assert 0.80 <= coverage <= 1.0
    assert result.supported is True


def test_switchback_whole_run_failure_counts_as_failed_not_a_crash(monkeypatch):
    """A coded refusal inside one replication's own pipeline is caught and
    counted as `failed`, not propagated to abort the entire run."""
    import increment.simulate.runner as runner_module
    from increment.errors import InvalidRequestError

    calls = {"n": 0}
    real = runner_module._run_switchback_replication

    def flaky(scenario):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise InvalidRequestError("boom", code="test.switchback.boom", context={})
        return real(scenario)

    monkeypatch.setattr(runner_module, "_run_switchback_replication", flaky)
    scenario = SwitchbackScenario(n_units=20, n_cycles=3, treatment_effect=2.0, seed=7)
    result = run_switchback_end_to_end(scenario, replications=4)
    assert result.attempted == 4
    assert result.failed == 2
    assert result.point_estimable == 2
    assert result.attempted == result.point_estimable + result.excluded + result.failed
    assert result.failure_reasons == {"test.switchback.boom": 2}


def test_switchback_json_serialization_emits_null_not_nan_for_a_single_replication():
    import json

    scenario = SwitchbackScenario(n_units=20, n_cycles=3, treatment_effect=2.0, seed=7)
    result = run_switchback_end_to_end(scenario, replications=1)
    payload = result.model_dump_json()
    assert "NaN" not in payload
    parsed = json.loads(payload)
    assert parsed["bias_mcse"]["outcome"] is None


def _valid_switchback_result_kwargs() -> dict[str, object]:
    return {
        "bias": {"outcome": 0.0},
        "bias_mcse": {"outcome": None},
        "coverage_conditional": {"outcome": 1.0},
        "coverage_conditional_mcse": {"outcome": None},
        "coverage_unconditional": {"outcome": 1.0},
        "coverage_unconditional_mcse": {"outcome": None},
        "attempted": 1,
        "point_estimable": 1,
        "interval_estimable": 1,
        "excluded": 0,
        "failed": 0,
        "failure_reasons": {},
        "exclusion_reasons": {},
        "interval_unavailable_reasons": {},
        "supported": True,
        "assumption_violation": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bias", {"other": 0.0}),
        ("attempted", 0),
        ("interval_estimable", 2),
        ("failed", -1),
        ("failure_reasons", {"boom": 1}),
        ("bias", {"outcome": float("nan")}),
        ("coverage_conditional", {"outcome": -0.01}),
        ("coverage_unconditional", {"outcome": None}),
        ("coverage_unconditional_mcse", {"outcome": 0.0}),
        ("bias_mcse", {"outcome": -0.01}),
    ],
)
def test_switchback_result_rejects_invalid_contracts(field, value):
    kwargs = _valid_switchback_result_kwargs()
    kwargs[field] = value

    with pytest.raises(ValidationError):
        SwitchbackEvalResult.model_validate(kwargs)


def test_switchback_result_rejects_finite_bias_with_no_estimable_points():
    kwargs = _valid_switchback_result_kwargs()
    kwargs.update(
        point_estimable=0,
        interval_estimable=0,
        excluded=1,
        exclusion_reasons={"excluded": 1},
        # Every other keyed aggregate already matches zero estimable points, so
        # the finite bias is the only contract left to refuse.
        coverage_conditional={"outcome": None},
        coverage_conditional_mcse={"outcome": None},
    )

    with pytest.raises(ValidationError) as caught:
        SwitchbackEvalResult.model_validate(kwargs)
    assert [(error["type"], error["loc"]) for error in caught.value.errors()] == [
        ("simulate.runner.result_contract", ())
    ]


def test_switchback_result_copies_and_freezes_mappings():
    kwargs = _valid_switchback_result_kwargs()
    bias = {"outcome": 0.0}
    kwargs["bias"] = bias
    result = SwitchbackEvalResult.model_validate(kwargs)

    bias["outcome"] = 99.0
    assert result.bias["outcome"] == 0.0
    with pytest.raises(TypeError):
        result.bias["outcome"] = 99.0  # type: ignore[index]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "field", ["failure_reasons", "exclusion_reasons", "interval_unavailable_reasons"]
)
def test_switchback_result_omitted_reason_defaults_are_frozen(field):
    kwargs = _valid_switchback_result_kwargs()
    del kwargs[field]
    result = SwitchbackEvalResult.model_validate(kwargs)

    with pytest.raises(TypeError):
        getattr(result, field)["new"] = 1  # type: ignore[index]


def test_supported_switchback_carryover_decays_during_washout_only():
    scenario = SwitchbackScenario(
        n_units=24,
        n_cycles=3,
        treatment_effect=0.0,
        temporal_effect=0.0,
        unit_effect_sd=0.0,
        washout_steps=3,
        observation_steps=2,
        seed=12,
    )
    panel = simulate_switchback_panel(scenario).to_pandas()
    values = panel.set_index(["unit", "cycle", "period", "step"])["outcome"]

    observed = False
    for unit in panel["unit"].unique():
        for cycle in range(scenario.n_cycles):
            prior_group = panel.loc[
                (panel["unit"] == unit) & (panel["cycle"] == cycle) & (panel["period"] == 0),
                "group",
            ].iloc[0]
            current_group = panel.loc[
                (panel["unit"] == unit) & (panel["cycle"] == cycle) & (panel["period"] == 1),
                "group",
            ].iloc[0]
            if prior_group != "treatment":
                continue
            observed = True
            washout = [values.loc[(unit, cycle, 1, step)] for step in range(scenario.washout_steps)]
            admitted = [
                values.loc[(unit, cycle, 1, step)]
                for step in range(
                    scenario.washout_steps,
                    scenario.washout_steps + scenario.observation_steps,
                )
            ]
            assert current_group == "control"
            assert washout[0] > washout[1] > washout[2] > 0.0
            assert admitted == [0.0, 0.0]
    assert observed


@pytest.mark.parametrize(
    ("allocation", "expected_context"),
    [
        (
            {"control": 0.4, "treatment": 0.5},
            {
                "message": "switchback allocation weights must sum to one",
                "allocation": {"control": 0.4, "treatment": 0.5},
                "reason": "invalid_allocation",
            },
        ),
        (
            {"control": 0.4, "treatment": 0.6},
            {
                "message": "switchback allocation requires exact 0.5/0.5 control/treatment weights",
                "allocation": {"control": 0.4, "treatment": 0.6},
                "control_group": "control",
                "treatment_group": "treatment",
                "reason": "unequal_allocation",
            },
        ),
    ],
)
def test_switchback_invalid_allocation_refuses_before_frame_access(
    monkeypatch, allocation, expected_context
):
    from increment.errors import InvalidRequestError
    from increment.switchback import from_switchback_panel

    def fail(*_args, **_kwargs):
        raise AssertionError("frame access occurred before allocation validation")

    monkeypatch.setattr("increment.switchback.nw.from_native", fail)
    with pytest.raises(InvalidRequestError) as exc:
        from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"outcome": "mean"},
            identification=Randomized(
                control_group="control",
                allocation=allocation,
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
        )
    assert exc.value.code == "source.frame.switchback.identification"
    assert exc.value.context == expected_context


def test_switchback_sequence_probability_does_not_change_arm_allocation():
    panel = simulate_switchback_panel(
        SwitchbackScenario(
            n_units=8,
            n_cycles=2,
            probability_ct=0.75,
            unit_effect_sd=0.0,
            temporal_effect=0.0,
        )
    )
    from increment.switchback import from_switchback_panel

    source = from_switchback_panel(
        panel,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"outcome": "mean"},
        identification=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.75),
            window=SwitchbackWindow(washout_steps=1, observation_steps=2),
        ),
    )

    assert source.diagnostics.probability_ct == pytest.approx(0.75)
    assert source.context.study.identification.allocation == {
        "control": 0.5,
        "treatment": 0.5,
    }


def test_switchback_pre_frame_plan_refusals_are_coded(monkeypatch):
    from increment.errors import CapabilityError
    from increment.semantics.models import AnalysisPlan, ExperimentMetric, MethodSpec
    from increment.switchback import from_switchback_panel

    def fail(*_args, **_kwargs):
        raise AssertionError("frame access occurred before plan validation")

    monkeypatch.setattr("increment.switchback.nw.from_native", fail)
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="outcome",
            decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
        )
    )
    with pytest.raises(CapabilityError) as exc:
        from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"outcome": "mean"},
            identification=Randomized(
                control_group="control",
                allocation={"control": 0.5, "treatment": 0.5},
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
            plan=plan,
        )
    assert exc.value.code == "source.frame.switchback.plan"
    assert exc.value.context["reason"] == "unsupported_plan_override"


def test_switchback_scenario_shared_schedule_draws_one_order_per_cycle_for_whole_roster():
    scenario = SwitchbackScenario(
        n_units=5,
        n_cycles=6,
        treatment_effect=0.0,
        probability_ct=0.8,
        shared_schedule=True,
        seed=11,
    )
    panel = simulate_switchback_panel(scenario).to_pylist()
    for cycle in range(scenario.n_cycles):
        group_by_period: dict[int, str] = {}
        mismatched = False
        for row in panel:
            if row["cycle"] != cycle:
                continue
            key = row["period"]
            if key in group_by_period and group_by_period[key] != row["group"]:
                mismatched = True
            group_by_period[key] = row["group"]
        assert not mismatched, f"cycle {cycle} did not share one order across the roster"

    first = simulate_switchback_panel(scenario)
    second = simulate_switchback_panel(scenario)
    assert first.to_pylist() == second.to_pylist()


def test_switchback_scenario_default_path_unaffected_by_shared_schedule_field():
    """Adding shared_schedule=False must not perturb the existing rng draw order."""
    scenario = SwitchbackScenario(n_units=8, n_cycles=4, treatment_effect=1.0, seed=3)
    shared_default = scenario.model_copy(update={"shared_schedule": False})
    assert (
        simulate_switchback_panel(scenario).to_pylist()
        == simulate_switchback_panel(shared_default).to_pylist()
    )


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_shared_schedule_block_t_refuses_degenerate_cycle_order_and_recovers_coverage():
    """A skewed probability_ct with few blocks has a real chance every block
    realizes the same cycle order; block-t's between-block variance then
    measures only observation noise, not the treatment contrast, and the
    interval is confidently wrong. Construction must refuse those
    replications rather than silently misreport them: measured at these
    parameters (3 units, 6 blocks, probability_ct=0.8), about 26% of
    replications are degenerate and, before the admission fix, every one of
    them fails to cover -- pulling unconditional coverage down to ~0.71-0.74
    (matching 1 - 0.8**6 exactly). After the fix those replications are
    refused (not silently wrong), and coverage conditional on an interval
    actually being produced recovers to nominal.
    """
    scenario = SwitchbackScenario(
        n_units=3,
        n_cycles=6,
        washout_steps=1,
        observation_steps=3,
        probability_ct=0.8,
        temporal_effect=3.0,
        treatment_effect=0.0,
        noise_sd=1.0,
        unit_effect_sd=0.0,
        shared_schedule=True,
        seed=2026,
    )
    result = run_switchback_end_to_end(scenario, replications=400)
    assert result.attempted == 400
    # ~26.2% of replications realize every block with the same order
    # (0.8**6 + 0.2**6); each is now refused rather than silently wrong.
    assert 60 <= result.failed <= 170, result.failure_reasons
    assert result.failure_reasons.keys() == {"source.frame.switchback.schedule"}
    assert result.interval_estimable > 0
    coverage = result.coverage_conditional["outcome"]
    assert coverage is not None
    assert coverage >= 0.85


def test_shared_schedule_block_t_refuses_degenerate_cycle_order_smoke():
    """Fast smoke: the refusal mechanism fires at all, without the
    parameter_recovery tier's replication count."""
    scenario = SwitchbackScenario(
        n_units=3,
        n_cycles=6,
        washout_steps=1,
        observation_steps=3,
        probability_ct=0.8,
        temporal_effect=3.0,
        treatment_effect=0.0,
        noise_sd=1.0,
        unit_effect_sd=0.0,
        shared_schedule=True,
        seed=2026,
    )
    result = run_switchback_end_to_end(scenario, replications=40)
    assert result.attempted == 40
    assert result.failed >= 1
    assert result.failure_reasons.keys() <= {"source.frame.switchback.schedule"}
