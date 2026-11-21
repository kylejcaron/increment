"""The two assignment logs belong to the spillover plan, not experiment builders."""

import pytest
from pydantic import TypeAdapter, ValidationError

from increment.errors import InvalidRequestError
from increment.semantics.assignment import Assignment, ParallelAssignment
from increment.spillover import SpilloverDesign, SpilloverGroup, plan_spillover
from tests.capability_consumers_auxiliary import assert_expected


def test_ordinary_and_two_stage_assignments_are_not_interchangeable():
    ordinary = ParallelAssignment()
    with pytest.raises(InvalidRequestError) as error:
        plan_spillover(ordinary, meaningful_effect=0.05)  # ty: ignore[invalid-argument-type]
    assert error.value.code == "spillover.design.required"
    groups = tuple(SpilloverGroup(g, tuple(f"{g}{j}" for j in range(4)), 1, 2) for g in "AB")
    two_stage = SpilloverDesign(groups, {"all": 1}, "purchased", "one week")
    with pytest.raises(ValidationError):
        TypeAdapter(Assignment).validate_python(two_stage)
    for ingress in (
        "from_unit_summary",
        "from_unit_panel",
        "from_switchback_panel",
        "from_definitions",
        "from_unit_day_artifact",
        "from_moments",
    ):
        assert_expected(
            "spillover_family_boundary", f"{ingress}_not_expressible", kind="not_expressible"
        )
