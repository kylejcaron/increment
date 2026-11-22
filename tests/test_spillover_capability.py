"""A two-stage offer design needs both assignment logs, not an ordinary arm label."""

import pytest
from pydantic import TypeAdapter, ValidationError

from increment import from_definitions, run, runnable
from increment.errors import CodedError
from increment.semantics.assignment import Assignment
from tests.capability_consumers_auxiliary import assert_expected, assert_support
from tests.parity_harness.matrix import SPILLOVER_DISPOSITIONS
from tests.spillover_trial import (
    frame_design,
    run_three_policy_probe,
    three_policy_trial,
    warehouse_run,
    write_study,
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
def test_families_share_a_definitions_directory(example_definitions, con):
    from contextlib import ExitStack

    design, frames = three_policy_trial()
    names = {role: f"mixed_families_{role}" for role in frames}
    write_study(example_definitions / "offers.yaml", design, names)
    with ExitStack() as stack:
        for role in ("roster", "offer_counts"):
            con.create_table(names[role], obj=frames[role])
            stack.callback(con.drop_table, names[role])
        ordinary, reader = from_definitions(
            example_definitions, con, experiment_name="new_onboarding_v2"
        )
        stack.enter_context(reader)
        trial, spillover_reader = from_definitions(
            example_definitions, con, family="spillover", experiment_name="pilot"
        )
        stack.enter_context(spillover_reader)
        assert ordinary.experiment_id == "new_onboarding_v2"
        assert [p.policy for p in trial.planning] == ["middle", "upper"]


def test_ordinary_preflight_refuses_spillover_before_observations():
    from increment.spillover import from_frame

    design, frames = three_policy_trial()
    trial, reader = from_frame(
        frames["roster"], offer_counts=frames["offer_counts"], design=frame_design(design)
    )
    with reader, pytest.raises(CodedError) as exc:
        runnable(trial, "bought")  # ty: ignore[invalid-argument-type]  # Unsupported family.
    assert exc.value.code == "kernel.request.invalid"


@pytest.mark.integration
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_named_spillover_comparisons_match_through_frame_and_definitions(con, tmp_path, alpha):
    from increment.spillover import from_frame

    design, frames = three_policy_trial(alpha=alpha)
    exp, reader = from_frame(**frames, design=frame_design(design))
    with reader:
        local = run(exp, reader)
    warehouse = run_three_policy_probe(con, tmp_path, alpha=alpha)
    assert_support("spillover_family_boundary", "dataframe_supported", local)
    assert_support("spillover_family_boundary", "warehouse_supported", warehouse)
    for frame, db in zip(local.comparisons, warehouse.comparisons, strict=True):
        assert (frame.policy, frame.reference) == (db.policy, db.reference)
        assert frame.estimate == pytest.approx(db.estimate, abs=1e-14, rel=0)
        assert (frame.lower, frame.upper) == pytest.approx((db.lower, db.upper), abs=1e-10, rel=0)
        assert frame.direction == db.direction
        assert frame.unavailable_reason == db.unavailable_reason
        assert frame.method == exp.method == db.method
        assert (frame.family_size, frame.policy_count, frame.alpha) == (
            db.family_size,
            db.policy_count,
            db.alpha,
        )


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
def test_family_reports_direction_and_interval_for_constant_sibling(source, con, tmp_path):
    from calibration.spillover import three_levels
    from increment.spillover import from_frame

    design, frames = three_levels()
    if source == "frame":
        experiment, reader = from_frame(**frames, design=design)
        with reader:
            middle, upper = run(experiment, reader).comparisons
    else:
        middle, upper = warehouse_run(con, tmp_path, design.declaration(), frames).comparisons
    # Middle has 139 successes among 300 binary group rates, upper has none
    # of 300 and base none of 400. The tail product is (1-t)^(n-s) *
    # (1-t+t/m)^s for s successes; average over dyadic bets and invert
    # at 6/.05 for each arm and its complement.
    assert middle.estimate == pytest.approx(139 / 300)
    assert (middle.lower, middle.upper) == pytest.approx(
        (0.338377423607, 0.571194044052), abs=1e-10
    )
    assert middle.direction == "increase"
    assert upper.estimate == 0
    assert (upper.lower, upper.upper) == pytest.approx((-0.018367258454, 0.024410113044), abs=1e-10)
    assert upper.direction == "inconclusive"
    assert middle.family_size == upper.family_size == 2
