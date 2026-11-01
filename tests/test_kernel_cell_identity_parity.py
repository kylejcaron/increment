from __future__ import annotations

from typing import Any

import pytest

from increment import Method, Scope, from_definitions, from_switchback_frame
from increment.declarations import CellKey
from increment.kernel.contracts import CellIdentity
from tests.analysis_factory import view_readout


def _contrast_analysis() -> Any:
    import pandas as pd

    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized
    from increment.semantics.unit_cycle import UnitCycleTApproximation

    rows = []
    for unit, first_order in (("u1", "control"), ("u2", "treatment")):
        for cycle in range(4):
            first = (
                first_order
                if cycle % 2 == 0
                else ("treatment" if first_order == "control" else "control")
            )
            order = (first, "treatment" if first == "control" else "control")
            for period, group in enumerate(order):
                for step in range(2):
                    value = float(cycle + period + step + (group == "treatment"))
                    rows.append(
                        {
                            "unit": unit,
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": group,
                            "value_a": value,
                            "value_b": value + 1,
                            "value_c": value + 2,
                        }
                    )
    metrics = {"value_a": "mean", "value_b": "mean", "value_c": "mean"}
    return from_switchback_frame(
        pd.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics=metrics,
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=1, observation_steps=1),
        ),
        contrast_references={metric: UnitCycleTApproximation() for metric in metrics},
    )


@pytest.mark.integration
@pytest.mark.slow
def test_cell_identity_from_row_agrees_with_cell_key_for_readout_rows(
    seeded_con: Any, seeded_defs: str
):
    analysis_experiment, analysis_resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    contrast_experiment, contrast_resolver = _contrast_analysis()
    rows_by_kind = {
        "arm": view_readout(
            analysis_experiment, analysis_resolver, decision_method=Method(name="unadjusted")
        ),
        "daily": view_readout(
            analysis_experiment, analysis_resolver, Scope(view="daily", grain="daily", values=True)
        ),
        "breakout": view_readout(
            analysis_experiment,
            analysis_resolver,
            Scope(view="breakout", grain="total"),
            decision_method=Method(name="unadjusted"),
        ),
        "contrast": view_readout(contrast_experiment, contrast_resolver),
    }
    checked: set[str] = set()
    for kind, rows in rows_by_kind.items():
        for row in rows:
            identity = CellIdentity.from_row(row)
            assert identity.key == CellKey.from_row(row)
            assert identity.source_snapshot_id == getattr(row, "source_snapshot_id", None)
            checked.add(kind)
    assert {"arm", "daily", "breakout", "contrast"} <= checked
