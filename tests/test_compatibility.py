import pytest

import increment.compatibility as compatibility
from increment.compatibility import Supported, UnitFloor, Unsupported


def test_supported_assumptions_coerced_to_tuple():
    """assumptions is declared tuple[str, ...]; a caller-supplied list must
    not silently become the runtime type, or a caller-held reference to
    that list could mutate a validated Supported after construction."""
    caller_list = ["independent_units", "cluster_robust"]
    s = Supported(
        assumptions=caller_list,  # ty: ignore[invalid-argument-type]
        floor=UnitFloor(minimum_per_arm=2),
        reference="ref",
    )
    assert isinstance(s.assumptions, tuple)
    assert s.assumptions == ("independent_units", "cluster_robust")

    caller_list.append("mutated_after_construction")
    assert s.assumptions == ("independent_units", "cluster_robust")


def test_supported_assumptions_frozen_after_construction():
    s = Supported(
        assumptions=("independent_units",),
        floor=UnitFloor(minimum_per_arm=2),
        reference="ref",
    )
    with pytest.raises(AttributeError):
        s.assumptions = ("mutated",)  # ty: ignore[invalid-assignment]


def test_unsupported_is_still_constructible():
    u = Unsupported(refusal_code="contrast.decision")
    assert u.refusal_code == "contrast.decision"


def test_arm_request_models_are_not_reexported_from_low_level_compatibility():
    assert not hasattr(compatibility, "ArmCompatibilityRequest")
    assert not hasattr(compatibility, "ArmPlanningProcedure")
    assert not hasattr(compatibility, "ContrastCompatibilityRequest")
