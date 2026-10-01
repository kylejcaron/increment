from __future__ import annotations

import pytest

from increment.analysis import Analysis
from increment.errors import CapabilityError
from increment.estimation.engine import Method
from increment.semantics.design import Randomized
from tests.test_readouts_switchback import _assignment, _frame


def _analysis():
    return Analysis.from_switchback_panel(
        _frame(),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"value": "mean"},
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=_assignment(),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_method": Method(name="cuped", variance_reduction="cuped")},
        {"sensitivity_methods": ()},
        {"prior": None},
    ],
)
def test_switchback_rejects_any_explicit_arm_override_before_stats(kwargs):
    analysis = _analysis()
    with pytest.raises(CapabilityError) as exc:
        analysis.run(**kwargs)
    assert exc.value.code == "readout.contrast.override"
    assert exc.value.context["argument"] in kwargs
