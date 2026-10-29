from __future__ import annotations

from increment.decide import decide


def test_decision_evaluator_is_a_public_package_api():
    assert callable(decide)
