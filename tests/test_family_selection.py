from __future__ import annotations

from increment import readouts
from increment.estimation.engine import Method
from tests.test_readouts_plan import _cuped_family_source


def test_family_uses_only_explicit_decision_rows():
    source = _cuped_family_source()
    results = readouts.run(
        source,
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        sensitivity_methods=(Method(name="unadjusted"),),
    )

    sensitivity = [row for row in results if row.method_role == "sensitivity"]
    assert sensitivity
    assert all(row.discovery is None for row in sensitivity)
    assert all(row.family_axes is None for row in sensitivity)
    assert all(row.family_q is None for row in sensitivity)
    assert all(row.family_threshold is None for row in sensitivity)
