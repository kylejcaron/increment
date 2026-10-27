from __future__ import annotations

from typing import Any

import pytest

from increment import Analysis, Method
from increment.declarations import CellKey
from increment.kernel.contracts import CellIdentity


def _contrast_analysis() -> Analysis:
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
    return Analysis.from_switchback_panel(
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
    analysis = Analysis("new_onboarding_v2", seeded_defs, seeded_con)
    contrast_analysis = _contrast_analysis()
    rows_by_kind = {
        "arm": analysis.run(decision_method=Method(name="unadjusted")),
        "daily": analysis.run_daily(),
        "breakout": analysis.run_breakout(decision_method=Method(name="unadjusted")),
        "contrast": contrast_analysis.run(),
    }
    checked: set[str] = set()
    for kind, rows in rows_by_kind.items():
        for row in rows:
            identity = CellIdentity.from_row(row)
            assert identity.key == CellKey.from_row(row)
            assert identity.source_snapshot_id == getattr(row, "source_snapshot_id", None)
            checked.add(kind)
    assert {"arm", "daily", "breakout", "contrast"} <= checked
