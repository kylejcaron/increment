"""A two-stage offer design needs both assignment logs, not an ordinary arm label."""

from contextlib import ExitStack
from inspect import signature

import pyarrow as pa
import pytest
import yaml

from increment import (
    MetricSpec,
    RunResult,
    Spillover,
    SpilloverComparison,
    from_definitions,
    from_frame,
    from_moments,
    from_switchback_frame,
    from_unit_day_artifact,
    run,
    runnable,
)
from increment.errors import CodedError
from increment.power import spillover_power
from increment.results import SpilloverEffects
from tests.capability_consumers_auxiliary import assert_expected, assert_support
from tests.spillover_trial import (
    frame_design,
    frame_trial,
    run_three_policy_probe,
    three_policy_trial,
    warehouse_run,
    write_study,
)


def _comparisons(result: RunResult) -> SpilloverEffects:
    rows = result.readout()
    assert isinstance(rows, SpilloverEffects)
    return rows


def test_other_inputs_cannot_declare_both_random_draws():
    design = Spillover(
        control_group="no_offer",
        offered_group="offered",
        groups_per_policy={"all": {"few": 2, "many": 2}},
        comparisons=(SpilloverComparison("many", "few", 0.05),),
    )
    panel = pa.table({"person": ["a"], "day": [1], "arm": ["no_offer"], "bought": [0]})
    with pytest.raises(CodedError) as refused:
        from_frame(
            panel,
            shape="panel",
            unit="person",
            date="day",
            group="arm",
            metrics=[MetricSpec(name="bought", type="conversion", window_days=7)],
            design=design,  # ty: ignore[invalid-argument-type]  # Test a summary-only design.
        )
    assert refused.value.code == "kernel.request.invalid"
    assert_expected("spillover", "from_unit_panel_not_expressible", kind="not_expressible")

    for ingress, builder in (
        ("from_switchback_panel", from_switchback_frame),
        ("from_moments", from_moments),
        ("from_unit_day_artifact", from_unit_day_artifact),
    ):
        # These constructors have no input for the first draw or the per-policy quotas.
        assert {"offer_counts", "group_policy"}.isdisjoint(signature(builder).parameters)
        assert_expected("spillover", f"{ingress}_not_expressible", kind="not_expressible")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mistake", "field"),
    [
        ("two_metrics", "plan"),
        ("mean", "metric"),
        ("missing_region", "stratum"),
        ("same_unit_store", "relations"),
        ("same_region_store", "relations"),
        ("same_region_unit", "relations"),
    ],
)
def test_both_loaders_refuse_the_same_declaration_before_reading(con, tmp_path, mistake, field):
    design, frames = three_policy_trial()
    cluster = "person_id" if mistake == "same_unit_store" else "group_id"
    stratum = {
        "missing_region": None,
        "same_region_store": cluster,
        "same_region_unit": "person_id",
    }.get(mistake, "stratum")
    if mistake == "missing_region":
        quotas = design["groups_per_policy"]["all"]
        design["groups_per_policy"] = {"north": quotas, "south": quotas}
    design.update(cluster=cluster, stratum=stratum)
    path = tmp_path / "trial.yaml"
    write_study(path, design, {key: key for key in frames})
    catalog = yaml.safe_load(path.read_text())
    metric = catalog["metrics"][0]
    specs = [
        MetricSpec(name="bought", type="mean" if mistake == "mean" else "conversion", window_days=7)
    ]
    if mistake == "two_metrics":
        catalog["metrics"].append(metric | {"name": "other"})
        catalog["experiments"][0]["plan"]["primary"] = ["bought", "other"]
        specs.append(MetricSpec(name="other", type="conversion", window_days=7))
    if mistake == "mean":
        metric.update(type="mean", aggregation="count")
    path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(CodedError) as warehouse:
        from_definitions(path, con, experiment_name="pilot")
    with pytest.raises(CodedError) as local:
        from_frame(
            frames["roster"],
            shape="summary",
            unit="person_id",
            group="assignment",
            cluster=cluster,
            stratum=stratum,
            design=frame_design(design),
            metrics=specs,
            offer_counts=frames["offer_counts"],
        )
    assert local.value.code == warehouse.value.code == "spillover.design.invalid"
    assert local.value.context["field"] == warehouse.value.context["field"] == field


@pytest.mark.integration
def test_named_experiments_share_a_definitions_directory(example_definitions, con):
    design, frames = three_policy_trial()
    names = {role: f"store_offers_{role}" for role in frames if role != "summary"}
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
            example_definitions, con, experiment_name="pilot"
        )
        stack.enter_context(spillover_reader)
        assert ordinary.experiment_id == "new_onboarding_v2"
        assert [p.policy for p in spillover_power(trial)] == ["middle", "upper"]


def test_preflight_defers_assignment_checks_without_reading_outcomes():
    trial, reader = frame_trial(*three_policy_trial(), planning=True)
    with reader:
        methods = runnable(trial, "bought")
    assert set(methods) == {"spillover.two_stage"}
    assert not methods["spillover.two_stage"].static_refusals
    assert methods["spillover.two_stage"].deferred_checks


@pytest.mark.integration
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_named_spillover_comparisons_match_through_frame_and_definitions(con, tmp_path, alpha):
    design, frames = three_policy_trial(alpha=alpha)
    exp, reader = frame_trial(design, frames)
    with reader:
        local = run(exp, reader)
    warehouse = run_three_policy_probe(con, tmp_path, alpha=alpha)
    assert_support("spillover", "from_unit_summary_supported", local)
    assert_support("spillover", "from_definitions_supported", warehouse)
    for frame, db in zip(_comparisons(local), _comparisons(warehouse), strict=True):
        assert (frame.metric, frame.window_days, frame.policy, frame.reference) == (
            db.metric,
            db.window_days,
            db.policy,
            db.reference,
        )
        assert frame.estimate == pytest.approx(db.estimate, abs=1e-14, rel=0)
        assert (frame.lower, frame.upper) == pytest.approx((db.lower, db.upper), abs=1e-10, rel=0)
        assert frame.direction == db.direction
        assert frame.unavailable_reason == db.unavailable_reason
        assert frame.range_basis == db.range_basis
        assert frame.method == db.method == "two-stage-fixed-bet-v3"
        assert (frame.family_size, frame.policy_count, frame.alpha) == (
            db.family_size,
            db.policy_count,
            db.alpha,
        )


@pytest.mark.integration
def test_both_public_paths_reject_missing_store_assignments(con, tmp_path):
    design, frames = three_policy_trial()
    frames["group_policy"] = frames["group_policy"].slice(1)
    exp, reader = frame_trial(design, frames)
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

    design, frames = three_levels()
    if source == "frame":
        experiment, reader = frame_trial(design, frames)
        with reader:
            result = run(experiment, reader)
    else:
        result = warehouse_run(con, tmp_path, design, frames)
    middle, upper = _comparisons(result)
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
