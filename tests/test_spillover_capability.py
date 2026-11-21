"""A two-stage offer design needs both assignment logs, not an ordinary arm label."""

from math import sqrt
from statistics import NormalDist

import pytest
from pydantic import TypeAdapter, ValidationError

from increment import run
from increment.errors import CodedError
from increment.semantics.assignment import Assignment
from tests.capability_consumers_auxiliary import assert_expected, assert_support
from tests.parity_harness.matrix import SPILLOVER_DISPOSITIONS
from tests.spillover_trial import (
    frame_design,
    run_three_policy_probe,
    three_policy_trial,
    warehouse_run,
)


def test_ordinary_and_two_stage_assignments_are_not_interchangeable():
    design, _ = three_policy_trial()
    with pytest.raises(ValidationError):
        TypeAdapter(Assignment).validate_python(frame_design(design))
    for ingress in (
        i for i, status in SPILLOVER_DISPOSITIONS.items() if status == "not_expressible"
    ):
        assert_expected(
            "spillover_family_boundary", f"{ingress}_not_expressible", kind="not_expressible"
        )


@pytest.mark.integration
def test_named_spillover_comparisons_match_through_frame_and_definitions(con, tmp_path):
    from increment.spillover import from_frame

    design, frames = three_policy_trial()
    exp, reader = from_frame(**frames, design=frame_design(design))
    with reader:
        local = run(exp, reader)
    warehouse = run_three_policy_probe(con, tmp_path)
    assert_support("spillover_family_boundary", "dataframe_supported", local)
    assert_support("spillover_family_boundary", "warehouse_supported", warehouse)
    for frame, db in zip(local.comparisons, warehouse.comparisons, strict=True):
        assert (frame.policy, frame.reference) == (db.policy, db.reference)
        assert frame.estimate == pytest.approx(db.estimate, abs=1e-14, rel=0)
        assert (frame.lower, frame.upper, frame.observed_reason) == (
            db.lower,
            db.upper,
            db.observed_reason,
        )
        assert frame.qualification_reason == db.qualification_reason
        assert frame.family_size == db.family_size == 2


@pytest.mark.integration
def test_both_public_paths_reject_a_missing_untreated_outcome(con, tmp_path):
    from increment.spillover import from_frame

    design, frames = three_policy_trial()
    frames["outcomes"] = frames["outcomes"].slice(1)
    exp, reader = from_frame(**frames, design=frame_design(design))
    with reader, pytest.raises(CodedError) as local:
        run(exp, reader)
    with pytest.raises(CodedError) as warehouse:
        warehouse_run(con, tmp_path, design, frames)
    assert local.value.code == warehouse.value.code == "spillover.log.keys"
    assert local.value.context == warehouse.value.context


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("source", ["frame", "warehouse"])
def test_qualified_family_interval_keeps_its_missing_sibling(source, con, tmp_path):
    from calibration.spillover import three_levels
    from increment.spillover import from_frame

    design, frames = three_levels()
    if source == "frame":
        experiment, reader = from_frame(**frames, design=design)
        with reader:
            middle, upper = run(experiment, reader).comparisons
    else:
        middle, upper = warehouse_run(con, tmp_path, design.declaration(), frames).comparisons
    # 139 of 300 middle-rate groups have all successes, other groups all failures.
    # Equal-size groups make the person-weighted variance s²/G = p(1-p)/(G-1).
    p = 139 / 300
    radius = NormalDist().inv_cdf(0.9875) * sqrt(p * (1.0 - p) / 299)
    assert middle.estimate == pytest.approx(p)
    assert (middle.lower, middle.upper) == pytest.approx((p - radius, p + radius))
    assert upper.estimate == 0 and upper.observed_reason == "no_outcome_variation"
    assert upper.lower is upper.upper is None
    assert middle.family_size == upper.family_size == 2
