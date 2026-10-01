from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd
import pytest

from increment import (
    Analysis,
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.errors import InvalidRequestError
from increment.estimation.contrast_results import ContrastResult, ContrastResults
from increment.semantics.design import Randomized
from increment.semantics.unit_cycle import UnitCycleTApproximation
from increment.tables import contrast_results_to_readout
from tests.analysis_factory import contrast_rows


def test_public_switchback_surface_smoke():
    rows: list[dict[str, object]] = []
    for unit, order in (("u1", ("control", "treatment")), ("u2", ("treatment", "control"))):
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": float(step + (group == "treatment")),
                        }
                    )
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        contrast_references={"value": UnitCycleTApproximation()},
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    result = contrast_rows(analysis.run())[0]
    assert math.isfinite(result.estimate.value)
    assert analysis.assignment_diagnostic().n_units == 2


def test_zero_variance_contrast_is_not_stat_sig():
    """Four units with deterministic, zero-variance per-unit deltas produce
    a zero standard error; the readout must not claim significance for a
    contrast whose typed evidence is unavailable."""
    rows: list[dict[str, object]] = []
    for i, unit in enumerate(("u1", "u2", "u3", "u4")):
        order = ("control", "treatment") if i % 2 == 0 else ("treatment", "control")
        for cycle in range(2):
            for period, group in enumerate(order):
                for step in range(2):
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": 10.0 if group == "treatment" else 5.0,
                        }
                    )
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        contrast_references={"value": UnitCycleTApproximation()},
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    result = contrast_rows(analysis.run())[0]
    assert result.standard_error == 0.0
    assert result.estimate.lb is None and result.estimate.ub is None
    row = contrast_results_to_readout([result])[0]
    assert row["stat_sig"] is False


def _shared_panel_rows(roster, block_order, outcome_fn, *, washout=1, observation=1):
    rows: list[dict[str, object]] = []
    total_steps = washout + observation
    for unit in roster:
        for cycle, order in block_order.items():
            for period, group in enumerate(order):
                for step in range(total_steps):
                    value = 999.0 if step < washout else outcome_fn(unit, cycle, period, group)
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value": value,
                        }
                    )
    return rows


def test_shared_schedule_public_surface_smoke():
    """A declared shared schedule runs end to end through the public
    ``Analysis`` surface: block-level method/reference/dof, not unit-t."""
    roster = ["u1", "u2", "u3"]
    block_order = {
        0: ("control", "treatment"),
        1: ("control", "treatment"),
        2: ("treatment", "control"),
        3: ("treatment", "control"),
    }
    eps = {0: -1.0, 1: 1.0, 2: -2.0, 3: 2.0}

    def outcome_fn(unit, cycle, period, group):
        return 10.0 + (6.0 + eps[cycle]) * (group == "treatment")

    rows = _shared_panel_rows(roster, block_order, outcome_fn)
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
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
            sequence=SharedScheduleOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    result = contrast_rows(analysis.run())[0]
    assert math.isfinite(result.estimate.value)
    assert result.estimate.value == pytest.approx(6.0)
    assert result.method == "switchback_block_t"
    assert result.reference == "block_t"
    assert result.n_blocks == 4
    assert result.n_units == 3
    assert result.dof == 3.0
    # X_b = [5, 7, 4, 8]; sum of squared deviations is 10 across four blocks.
    from scipy.stats import t

    expected_se = math.sqrt(10.0 / (4 * 3))
    half_width = t.isf(0.025, 3) * expected_se
    assert result.standard_error == pytest.approx(expected_se)
    assert result.estimate.lb == pytest.approx(6.0 - half_width)
    assert result.estimate.ub == pytest.approx(6.0 + half_width)
    assert result.estimate.lb is not None and result.estimate.ub is not None
    assert result.estimate.lb < result.estimate.value < result.estimate.ub
    diagnostics = analysis.assignment_diagnostic()
    assert diagnostics.n_units == 3
    assert diagnostics.n_blocks == 4


def test_shared_schedule_zero_variance_contrast_is_not_stat_sig():
    """Acceptance: degenerate block spread reports unavailable uncertainty
    (zero SE, not statistically significant), never a fabricated width --
    no zero-width interval or confidence level, typed nulls through every
    output conversion."""
    import polars as pl

    roster = ["u1", "u2"]
    block_order = {
        0: ("control", "treatment"),
        1: ("treatment", "control"),
    }

    def outcome_fn(unit, cycle, period, group):
        return 10.0 if group == "treatment" else 5.0

    rows = _shared_panel_rows(roster, block_order, outcome_fn)
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
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
            sequence=SharedScheduleOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
    )
    results = contrast_rows(analysis.run())
    result = results[0]
    assert result.standard_error == 0.0
    estimate = result.estimate
    assert estimate.value == pytest.approx(5.0)
    assert (estimate.lb, estimate.ub, estimate.level, estimate.alpha) == (None, None, None, None)
    frame = results.to_frame(backend="polars")
    assert isinstance(frame, pl.DataFrame)
    assert frame.schema["lb"] == frame.schema["ub"] == pl.Float64
    assert frame["lb"].to_list() == frame["ub"].to_list() == [None]
    row = contrast_results_to_readout([result])[0]
    assert (row["lower"], row["higher"], row["level"]) == (None, None, None)
    assert row["stat_sig"] is False
    assert row["n_blocks"] == 2


@pytest.fixture
def degenerate_320_ct_panel():
    rows = []
    for u in range(80):
        unit_noise = (u - 39.5) * 1e-4
        for cycle in range(4):
            for period, arm in enumerate(("control", "treatment")):
                for step in (0, 1):
                    rows.append(
                        {
                            "unit": f"u{u}",
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "arm": arm,
                            "y": period * (1.0 + unit_noise),
                        }
                    )
    return pd.DataFrame(rows)


def test_degenerate_320_ct_witness_refuses_public_unit_cycle_declaration(
    degenerate_320_ct_panel,
):
    """The all-CT witness has 320 unit-cycle draws and degenerate contributions."""
    from scipy.stats import binomtest

    frame = degenerate_320_ct_panel
    draws = frame[["unit", "cycle", "period", "arm"]].drop_duplicates()
    ct = int((draws.loc[draws["period"] == 0, "arm"] == "control").sum())
    tc = int((draws.loc[draws["period"] == 0, "arm"] == "treatment").sum())
    assert (ct, tc) == (320, 0)
    assert binomtest(ct, ct + tc, 0.5).pvalue < 1e-6

    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis.from_switchback_panel(
            frame,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="arm",
            metrics={"y": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
        )
    error = exc_info.value
    assert error.code == "source.frame.switchback.schedule"
    assert error.context["reason"] == "implausible_realized_split"
    assert (error.context["ct_cycles"], error.context["tc_cycles"]) == (320, 0)
    assert error.context["probability_ct"] == 0.5
    with pytest.raises(TypeError):
        cast("dict[str, object]", error.context)["ct_cycles"] = 0


def test_degenerate_320_ct_witness_now_refuses_under_public_shared_declaration(
    degenerate_320_ct_panel,
):
    """Under the shared law this witness has only 4 block-level draws (one
    per declared cycle), not the 320 unit-cycle-inflated draws the
    unit-cycle sibling test sees: binomtest(4, 4, 0.5) is 0.125, nowhere
    near the 1e-6 mechanism threshold, so only the positivity check catches
    it. Before the fix this silently returned a "successful" SE=0.0 result
    -- an all-one-order shared panel whose between-block variance measures
    only observation noise, not the treatment contrast -- which is exactly
    the hazard this task refuses."""
    with pytest.raises(InvalidRequestError) as exc_info:
        Analysis.from_switchback_panel(
            degenerate_320_ct_panel,
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="arm",
            metrics={"y": "mean"},
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "treatment": 0.5}
            ),
            assignment=SwitchbackAssignment(
                sequence=SharedScheduleOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=1, observation_steps=1),
            ),
        )
    error = exc_info.value
    assert error.code == "source.frame.switchback.schedule"
    assert error.context["reason"] == "shared_schedule_missing_cycle_order"
    assert (error.context["ct_cycles"], error.context["tc_cycles"]) == (4, 0)
    assert error.context["n_blocks"] == 4
    assert isinstance(error.context["route"], str) and error.context["route"]


@pytest.mark.parametrize("carryover_order", [0, 1, 2])
@pytest.mark.parametrize("shared", [False, True], ids=["unit-cycle", "shared"])
def test_public_retained_window_metadata_and_effect_round_trip(shared, carryover_order):
    """Both laws retain declared steps and preserve their scale and provenance."""
    from increment.tables import estimates_to_readout

    roster = ["u1", "u2", "u3"]
    effects = {"u1": 4.0, "u2": 6.0, "u3": 8.0}
    block_order = {
        0: ("control", "treatment"),
        1: ("control", "treatment"),
        2: ("treatment", "control"),
        3: ("treatment", "control"),
    }
    eps = {0: -1.0, 1: 1.0, 2: -2.0, 3: 2.0}

    def outcome_fn(unit, cycle, period, group):
        return 10.0 + (effects[unit] + eps[cycle]) * (group == "treatment")

    rows = _shared_panel_rows(roster, block_order, outcome_fn, observation=3)
    for row in rows:
        if 1 <= cast(int, row["step"]) < 1 + carryover_order:
            row["value"] = 10000.0 * (row["group"] == "treatment")
    sequence = SharedScheduleOrder(probability_ct=0.5) if shared else IndependentBernoulliOrder()
    assignment = SwitchbackAssignment(
        sequence=sequence,
        window=SwitchbackWindow(
            washout_steps=1, observation_steps=3, carryover_order=carryover_order
        ),
    )
    restored_assignment = SwitchbackAssignment.model_validate_json(assignment.model_dump_json())
    assert type(restored_assignment.sequence) is type(sequence)
    analysis = Analysis.from_switchback_panel(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=restored_assignment,
        contrast_references=None if shared else {"value": UnitCycleTApproximation()},
    )
    results = analysis.run()
    result = contrast_rows(results)[0]
    retained = 3 - carryover_order
    expected_se = retained * (math.sqrt(10 / 12) if shared else math.sqrt(8 / 6))
    assert result.estimate.value == pytest.approx(6 * retained)
    assert result.standard_error == pytest.approx(expected_se)
    assert result.dof == (3 if shared else 2)
    assert result.estimate.lb is not None and result.estimate.ub is not None
    assert result.estimate.lb < result.estimate.value < result.estimate.ub
    assert ContrastResult.model_validate_json(result.model_dump_json()) == result

    diagnostic = analysis.assignment_diagnostic()
    assert diagnostic.n_cycles == 12
    assert (diagnostic.ct_count, diagnostic.tc_count) == ((2, 2) if shared else (6, 6))
    assert diagnostic.observation_rows == 72
    assert diagnostic.retained_rows == 24 * retained
    assert diagnostic.observation_steps == 3
    assert diagnostic.retained_steps == retained
    assert diagnostic.randomization_law == sequence.scheme
    assert diagnostic.independence_grain == sequence.independence_unit
    assert type(diagnostic).model_validate_json(diagnostic.model_dump_json()) == diagnostic
    frame = cast(Any, ContrastResults([result]).to_frame())
    frame_row = frame.iloc[0].to_dict()
    for row in (result.model_dump(), frame_row, estimates_to_readout(results)[0]):
        assert row["observation_steps"] == 3
        assert row["retained_steps"] == retained
        assert row["carryover_order"] == carryover_order
        assert row["randomization_law"] == sequence.scheme
        assert row["independence_grain"] == sequence.independence_unit
        assert row["n_units"] == 3
        assert row["n_cycles"] == 12
        assert row["estimand"] == "retained_window_total_difference"
        assert row["identifying_assumption"] == "no_residual_carryover_after_discarded_steps"
