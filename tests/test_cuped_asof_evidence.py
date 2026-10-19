from __future__ import annotations

import pyarrow as pa
import pytest

from increment import CupedCovariate, Method, MetricSpec, readouts
from increment.errors import CodedError
from increment.frame import from_unit_panel
from tests.test_readouts_plan import _cuped_asof_source


@pytest.mark.parametrize("scalar_decision", [True, False])
def test_asof_scalar_roles_match_standalone(scalar_decision):
    scalar = Method.cuped(name="adjusted")
    naive = Method.unadjusted()
    joint = readouts.asof_lift(
        _cuped_asof_source(),
        decision_method=scalar if scalar_decision else naive,
        sensitivity_methods=(naive if scalar_decision else scalar,),
    )
    standalone = readouts.asof_lift(
        _cuped_asof_source(), decision_method=scalar, sensitivity_methods=()
    )
    adjusted = {
        (row.metric, row.group_id, row.ds): row for row in joint if row.method == scalar.name
    }
    assert len(adjusted) == len(standalone) == 8
    for expected in standalone:
        actual = adjusted[expected.metric, expected.group_id, expected.ds]
        assert actual.method_role == ("decision" if scalar_decision else "sensitivity")
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se is not None
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )


def test_asof_named_scalar_refuses_before_unbound_acquisition(monkeypatch):
    source = from_unit_panel(
        pa.Table.from_pylist(
            [
                {
                    "unit": f"{arm}-{unit}",
                    "arm": arm,
                    "day": day,
                    "outcome": float(10 + unit + day + (arm == "treatment")),
                    "history": float(unit),
                }
                for arm in ("control", "treatment")
                for unit in range(8)
                for day in (0, 1)
            ]
        ),
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics=[
            MetricSpec(
                name="outcome",
                decision_method=Method.cuped(
                    covariates=CupedCovariate(name="history", kind="continuous")
                ),
            )
        ],
    )

    def unbound_moments(*args, **kwargs):
        pytest.fail("Named scalar requests must be admitted before moment acquisition")

    monkeypatch.setattr(source, "moments", unbound_moments)
    with pytest.raises(CodedError) as raised:
        readouts.asof_lift(source)
    assert raised.value.code == "readout.scalar_method_evidence"
    assert raised.value.context["route"] == "readouts.asof_lift"
