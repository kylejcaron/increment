from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from increment import (
    Analysis,
    IndependentBernoulliOrder,
    Randomized,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
    UnitCycleTApproximation,
)
from increment.errors import CapabilityError, InvalidRequestError


def _run(analysis: Analysis, **kwargs: Any):
    return analysis.run(**kwargs)


def _panel(
    *, shared: bool = False, value_scale: float = 1.0, treatment_shift: float = 0.0
) -> pd.DataFrame:
    rows = []
    orders = (
        (("control", "treatment"), ("control", "treatment"))
        if shared
        else (
            ("control", "treatment"),
            ("treatment", "control"),
        )
    )
    for unit, unit_order in zip(("u1", "u2"), orders, strict=True):
        for cycle in range(3):
            order = (
                (("control", "treatment") if cycle % 2 == 0 else ("treatment", "control"))
                if shared
                else unit_order
            )
            for period, group in enumerate(order):
                for step in range(3):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": (
                                float(cycle + period + step)
                                + treatment_shift * (group == "treatment")
                            )
                            * value_scale,
                        }
                    )
    return pd.DataFrame(rows)


def _analysis(
    *,
    shared: bool = False,
    value_scale: float = 1.0,
    treatment_shift: float = 0.0,
    include_other: bool = False,
    frame: pd.DataFrame | None = None,
) -> Analysis:
    sequence = (
        SharedScheduleOrder(probability_ct=0.5)
        if shared
        else IndependentBernoulliOrder(probability_ct=0.5)
    )
    assignment = SwitchbackAssignment(
        sequence=sequence, window=SwitchbackWindow(washout_steps=1, observation_steps=2)
    )
    panel = (
        frame
        if frame is not None
        else _panel(
            shared=shared,
            value_scale=value_scale,
            treatment_shift=treatment_shift,
        )
    )
    metrics = {"value": "mean"}
    if include_other:
        panel["other"] = panel["value"] * 2
        metrics["other"] = "mean"
    return Analysis.from_switchback_panel(
        panel,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=metrics,
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=assignment,
        contrast_references=(
            None if shared else {name: UnitCycleTApproximation() for name in metrics}
        ),
    )


def test_switchback_readout_returns_captured_typed_result():
    result = _run(_analysis())

    assert len(result) == 1
    assert result[0].metric == "value"
    assert result.metadata is not None
    assert result.metadata.coherence == "captured"
    assert result.metadata.effective_modes == {"value": "captured"}


def test_switchback_results_remain_stable_after_panel_mutation():
    panel = _panel()
    analysis = _analysis(frame=panel)
    first = _run(analysis)[0]
    panel.loc[panel["group"] == "treatment", "value"] += 20
    second = _run(analysis)[0]
    fresh = _run(_analysis(frame=panel))[0]

    assert first.estimate == second.estimate
    assert first.standard_error == second.standard_error
    assert fresh.estimate != first.estimate
    assert second.source_snapshot_id is not None


def test_switchback_undeclared_metric_refuses_by_exact_declaration():
    analysis = _analysis()
    mismatched = analysis.metrics[0].model_copy(update={"description": "different metric"})

    with pytest.raises(InvalidRequestError) as error:
        analysis.run(metrics=[mismatched])

    assert error.value.code == "kernel.context.undeclared_stream"
    assert error.value.context == {"stream": "metric"}


def test_switchback_triggered_population_refuses_before_readout():
    with pytest.raises(CapabilityError) as error:
        _analysis().run(population="triggered")

    assert error.value.code == "facade.analysis.trigger_unsupported"


def test_switchback_metric_option_refuses_with_public_code():
    with pytest.raises(CapabilityError) as error:
        _analysis().run(estimands=["itt"])

    assert error.value.code == "facade.analysis.contrast_unavailable"


def test_switchback_request_preserves_inference_grain():
    independent = _run(_analysis())[0]
    shared = _run(_analysis(shared=True))[0]

    assert independent.randomization_law == "independent_bernoulli_order"
    assert independent.independence_grain == "unit_cycle"
    assert independent.n_units == 2
    assert independent.n_cycles == 6
    assert independent.ct_cycles == 3
    assert independent.tc_cycles == 3
    assert shared.randomization_law == "shared_schedule"
    assert shared.independence_grain == "shared_block"
    assert shared.n_units == 2
    assert shared.n_blocks == 3
    assert shared.ct_cycles == 2
    assert shared.tc_cycles == 1
    for result in (independent, shared):
        assert result.washout_steps == 1
        assert result.observation_steps == 2
        assert result.retained_steps == 2
        assert result.carryover_order == 0
        assert result.reference in ("unit_t_approximation", "block_t")
        assert result.estimate.lb is not None
        assert result.estimate.ub is not None
        assert result.source_snapshot_id is not None


def test_switchback_neighboring_integer_oracle():
    from scipy.stats import t

    from tests.parity_harness.cases import _neighboring_integer_switchback_analysis

    (result,) = _run(_neighboring_integer_switchback_analysis())
    assert result.estimate.value == pytest.approx(1.5)
    assert result.standard_error == pytest.approx(0.5)
    radius = 0.5 * t.isf(result.alpha / 2, 1)

    assert result.estimate.lb == pytest.approx(1.5 - radius)
    assert result.estimate.ub == pytest.approx(1.5 + radius)
