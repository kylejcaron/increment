"""Public planning, registration and fixed-horizon untreated outcome comparisons."""

from __future__ import annotations

from collections import UserList
from dataclasses import replace
from datetime import timedelta
from typing import Any, cast

import pandas as pd
import pytest
import yaml

from increment import Experiment, Spillover, SpilloverComparison, from_definitions, run, workflows
from increment.errors import CodedError
from increment.power import spillover_power
from tests.spillover_trial import (
    ANCHOR,
    WAREHOUSE_ROLES,
    WINDOW_END,
    bind_summary,
    complete_receipt,
    launched_study,
    spillover_rows,
    three_policy_trial,
    write_study,
)


def _trial(groups=4, size=4, low=1, high=2, baseline=None):
    names = [f"g{i:03}" for i in range(groups)]
    policies = ["high" if i < groups // 2 else "low" for i in range(groups)]
    people = pd.DataFrame(
        [
            (f"{i}:{j}", g, "all", "offered" if j < (high if p == "high" else low) else "no_offer")
            for i, (g, p) in enumerate(zip(names, policies, strict=True))
            for j in range(size)
        ],
        columns=["person_id", "group_id", "stratum", "assignment"],
    )
    counts = pd.DataFrame(
        [(g, p, k) for g in names for p, k in (("low", low), ("high", high))],
        columns=["group_id", "policy", "offers"],
    )
    design = Spillover(
        control_group="no_offer",
        offered_group="offered",
        groups_per_policy={"all": {"low": groups // 2, "high": groups // 2}},
        comparisons=(
            SpilloverComparison(
                policy="high", reference="low", meaningful_effect=0.10, baseline_rate=baseline
            ),
        ),
    )
    return people, counts, design, pd.DataFrame({"group_id": names, "policy": policies})


def test_no_history_keeps_wide_intervals_even_if_all_outcomes_are_zero():
    people, counts, design, policies = _trial()
    people["ordered"] = people.person_id.isin({"0:2", "1:2", "1:3", "3:1"})
    experiment, reader = bind_summary(people, counts, policies, design, metric="ordered")
    with reader:
        (metric,) = workflows.metrics(experiment, reader)
        (result,) = spillover_rows(
            run(
                experiment,
                reader,
                metrics=UserList([metric]),
                views=experiment.views,
                coherence="captured",
            )
        )
    assert (result.policy, result.reference, result.family_size) == ("high", "low", 1)
    assert result.estimate == pytest.approx(7 / 12)
    assert result.lower is not None and result.upper is not None
    assert result.lower < 0 < result.upper
    assert result.direction == "inconclusive" and result.unavailable_reason is None
    (plan,) = spillover_power(experiment)
    assert (plan.status, plan.reason) == ("not_assessed", "missing_baseline_rate")
    assert (result.metric, result.window_days) == ("ordered", 7)
    _, reader = bind_summary(people.assign(ordered=0), counts, policies, design, metric="ordered")
    with reader:
        (zero,) = spillover_rows(run(experiment, reader))
    assert zero.estimate == 0 and zero.direction == "inconclusive"
    # Two zero groups in each arm. The complement solves the average of
    # (1 - t + t/m)^2 for t=.95,.475 at 4/.05; both arms have the same width.
    assert (zero.lower, zero.upper) == pytest.approx((-0.914078742665, 0.914078742665), abs=1e-10)
    assert zero.unavailable_reason is None


@pytest.mark.parametrize("changed", [False, True], ids=["identical", "disjoint"])
def test_range_from_offer_counts_is_not_evidence_of_a_direction(changed):
    people, counts, design, policies = _trial(groups=32, low=1, high=1)
    if changed:
        # Only the unassigned high policy of the last group offers to one more person.
        counts.loc[(counts.group_id == "g031") & (counts.policy == "high"), "offers"] = 2
    people["bought"] = people.person_id.str.split(":").str[0].astype(int) < 16
    experiment, reader = bind_summary(people, counts, policies, design)
    with reader:
        (row,) = spillover_rows(run(experiment, reader))
    assert row.estimate == (1 if changed else 0)
    cap = 1 / 32 if changed else 0
    assert (row.lower, row.upper) == (-cap, cap)
    assert row.direction == "inconclusive" and row.unavailable_reason is None
    assert row.range_basis == ("attainable_range" if changed else "identical_offer_counts")

    if not changed:
        people["bought"] = people.bought.astype("boolean")
        people.loc[people.person_id == "0:1", "bought"] = pd.NA
        _, reader = bind_summary(people, counts, policies, design)
        with reader, pytest.raises(CodedError) as missing:
            run(experiment, reader)
        assert missing.value.code == "spillover.log.keys"
        assert missing.value.context["field"] == "outcomes"


def test_direction_requires_a_present_interval_strictly_on_one_side_of_zero():
    people, counts, design, policies = _trial()
    experiment, reader = bind_summary(people.assign(bought=0), counts, policies, design)
    with reader:
        (row,) = spillover_rows(run(experiment, reader))

    def with_interval(lower, upper):
        return type(row).model_validate(
            {**row.model_dump(exclude={"direction"}), "lower": lower, "upper": upper}
        )

    for lower, upper, label in (
        (0.1, 0.4, "increase"),
        (-0.4, -0.1, "decrease"),
        (-0.1, 0.4, "inconclusive"),
        (0.0, 0.4, "inconclusive"),
        (-0.4, 0.0, "inconclusive"),
        (0.0, 0.0, "inconclusive"),
        (None, None, "unavailable"),
    ):
        assert with_interval(lower, upper).direction == label
    failed = with_interval(float("nan"), float("nan"))
    assert failed.lower is failed.upper is None
    assert (failed.direction, failed.unavailable_reason) == ("unavailable", "nonfinite_interval")
    assert failed.range_basis == "unavailable"
    assert (failed.policy, failed.reference, failed.estimate) == (
        row.policy,
        row.reference,
        row.estimate,
    )


@pytest.mark.slow
def test_underpowered_trial_still_reports_observed_interval():
    people, counts, design, policies = _trial(240, 20, 5, 10, 0.25)
    group_index = people.person_id.str.split(":").str[0].astype(int)
    people["bought"] = (group_index < 45) | group_index.between(120, 134)
    experiment, reader = bind_summary(people, counts, policies, design)
    assert spillover_power(experiment)[0].status == "underpowered"
    with reader:
        (result,) = spillover_rows(run(experiment, reader))
    # With binary group rates the moment bound is the exact product: each of
    # the 120 factors is 1-t or 1-t+t/m. Average over t=.95/2^j for
    # j=0,...,7, using 45 and 15 successes and both tails.
    assert result.estimate == pytest.approx(0.25)
    assert (result.lower, result.upper) == pytest.approx(
        (-0.025936740094, 0.494200502288), abs=1e-10
    )
    assert result.direction == "inconclusive"
    assert result.unavailable_reason is None


def test_registration_reopen_and_reader_lifecycle_before_observation():
    people, counts, design, _ = _trial()
    roster = people.drop(columns="assignment")
    experiment, reader = bind_summary(roster, counts, None, design)
    with reader:
        registered = Experiment.from_json(experiment.to_json())
        reordered, other = bind_summary(roster.iloc[::-1], counts.iloc[::-1], None, design)
        with other:
            assert registered.design["protocol"] == "spillover-fixed-count-v2"
            assert registered.registration == reordered.registration
        different, wrong = bind_summary(roster, counts, None, design, experiment_id="other")
        with wrong, pytest.raises(CodedError) as mismatch:
            run(different, reader)
        assert mismatch.value.code == "kernel.run.execution_unavailable"
        assert mismatch.value.context["field"] == "experiment_id"
    assert reader.closed
    with pytest.raises(CodedError) as closed:
        run(experiment, reader)
    assert closed.value.code == "kernel.run.reader_closed"
    swapped = roster.copy()
    swapped.loc[[0, 4], "group_id"] = swapped.loc[[4, 0], "group_id"].to_list()
    _, rebound = bind_summary(swapped, counts, None, design)
    with rebound, pytest.raises(CodedError) as stale:
        run(registered, rebound)
    assert stale.value.code == "kernel.run.execution_unavailable"
    assert stale.value.context["field"] == "registration"


def test_registration_binds_comparison_and_group_specific_offer_counts():
    people, counts, design, _ = _trial()
    roster = people.drop(columns="assignment")
    original, reader = bind_summary(roster, counts, None, design)
    reverse = SpilloverComparison(policy="low", reference="high", meaningful_effect=-0.10)
    with reader:
        registered = Experiment.from_json(original.to_json())
    for declaration, schedule, options in (
        (design, counts, {"metric": "clicked"}),
        (design, counts, {"window_days": 8}),
        (replace(design, comparisons=(reverse,)), counts, {}),
        (design, counts, {"alpha": 0.01}),
        (design, counts.assign(offers=counts.offers.where(counts.index != 0, 0)), {}),
    ):
        _, rebound = bind_summary(roster, schedule, None, declaration, **options)
        with rebound, pytest.raises(CodedError) as stale:
            run(registered, rebound)
        assert stale.value.code == "kernel.run.execution_unavailable"


def test_unsupported_run_options_refuse_before_missing_logs():
    people, counts, design, _ = _trial()
    pair = bind_summary(people.drop(columns="assignment"), counts, None, design)
    with pair[1]:
        for options in (
            {"metrics": []},
            {"metrics": ["not_registered"]},
            {"execution_context": object()},
            {"decision_log": object()},
            {"capture": object()},
            {"prior": None},
            {"coherence": "live"},
            {"decision_method": object()},
            {"sensitivity_methods": []},
            {"estimands": []},
            {"value_scale": {}},
            {"exploratory_metrics": []},
        ):
            with pytest.raises(CodedError) as refused:
                cast(Any, run)(*pair, **options)
            assert refused.value.code == "kernel.request.invalid", options
        with pytest.raises(CodedError) as empty:
            run(*pair, views=[])
        assert empty.value.code == "kernel.run.execution_unavailable"
        with pytest.raises(CodedError) as population:
            run(*pair, population="triggered")
        assert population.value.code == "kernel.request.population_scope_only"


def _measure_saved_trial(db, path, saved):
    import ibis

    registered = Experiment.from_json(saved)
    con = ibis.duckdb.connect(db, threads=1)
    try:
        _, reader = from_definitions(
            path,
            con,
            experiment_name=registered.experiment_id,
            source_snapshot_evidence=complete_receipt(),
        )
        with reader:
            return [row.estimate for row in spillover_rows(run(registered, reader))]
    finally:
        con.disconnect()


@pytest.mark.integration
def test_prelaunch_catalog_reopens_offline_and_saved_declaration_runs_in_a_new_process(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    import ibis

    design, frames = three_policy_trial()
    path = tmp_path / "pilot.yaml"
    catalog = tmp_path / "frozen.json"
    db = tmp_path / "study.duckdb"
    write_study(path, design, {role: role for role in WAREHOUSE_ROLES}, dialect="duckdb")
    con = ibis.duckdb.connect(db)
    try:
        for role in ("roster", "offer_counts"):
            con.create_table(role, frames[role])
        experiment, reader = from_definitions(
            path, con, experiment_name="pilot", catalog_path=catalog
        )
        with reader:
            assert workflows.experiment_definition(experiment, reader).name == "pilot"
            assert [m.name for m in workflows.metrics(experiment, reader)] == ["bought"]
            assert workflows.available_metrics(experiment, reader) == []
            saved = experiment.to_json()
        for role in ("group_policy", "assignments", "purchases"):
            con.create_table(role, frames[role])
    finally:
        con.disconnect()
    reopened, offline = from_definitions(path, experiment_name="pilot", catalog_path=catalog)
    with offline:
        assert reopened.to_json() == saved
        assert [(p.policy, p.reason) for p in spillover_power(reopened)] == [
            ("middle", "missing_baseline_rate"),
            ("upper", "missing_baseline_rate"),
        ]
        with pytest.raises(CodedError) as unavailable:
            run(reopened, offline)
    assert unavailable.value.code == "kernel.run.reader_offline"
    # Unrelated additions may refresh the mutable catalog but cannot change this trial.
    definitions = yaml.safe_load(path.read_text())
    definitions["metrics"].append(
        {**definitions["metrics"][0], "name": "other_window", "window_days": 3}
    )
    path.write_text(yaml.safe_dump(definitions))
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as workers:
        points = workers.submit(_measure_saved_trial, db, path, saved).result()
    assert points == pytest.approx([0.030158730158730190, 0.480952380952380909], rel=0, abs=1e-14)


@pytest.mark.integration
def test_selection_and_incompatible_source_options_refuse_before_reading_trial_logs(con, tmp_path):
    design, _ = three_policy_trial()
    path = tmp_path / "studies.yaml"
    write_study(path, design, {role: role for role in WAREHOUSE_ROLES})
    catalog = yaml.safe_load(path.read_text())
    catalog["experiments"].append({**catalog["experiments"][0], "name": "replication"})
    path.write_text(yaml.safe_dump(catalog))
    with pytest.raises(CodedError) as ambiguous:
        from_definitions(path, con)
    assert ambiguous.value.code == "kernel.request.invalid"
    with pytest.raises(CodedError) as unknown:
        from_definitions(path, con, experiment_name="not-there")
    assert unknown.value.code == "facade.analysis.unknown_experiment"
    assert unknown.value.context["available"] == ("pilot", "replication")
    for options in (
        {"on_mixed_assignment": "exclude"},
        {"store": "none"},
        {"backend": "snowflake"},
    ):
        with pytest.raises(CodedError) as incompatible:
            cast(Any, from_definitions)(path, con, experiment_name="pilot", **options)
        assert incompatible.value.code == "kernel.request.invalid"


@pytest.mark.integration
def test_saved_registration_refuses_changed_purchase_or_assignment_definition_before_logs(tmp_path):
    import ibis

    design, frames = three_policy_trial()
    path = tmp_path / "pilot.yaml"
    write_study(path, design, {role: role for role in WAREHOUSE_ROLES}, dialect="duckdb")
    original = path.read_text()
    con = ibis.duckdb.connect()
    try:
        for role in ("roster", "offer_counts"):
            con.create_table(role, frames[role])
        experiment, reader = from_definitions(path, con, experiment_name="pilot")
        with reader:
            registered = Experiment.from_json(experiment.to_json())
        source = yaml.safe_load(original)
        for section, key, replacement in (
            ("metrics", "window_days", 8),
            ("metrics", "filters", [{"property": "channel", "op": "equals", "values": ["shop"]}]),
            ("fact_sources", "sql", source["fact_sources"][0]["sql"] + " WHERE channel <> 'web'"),
            ("exposures", "sql", source["exposures"][0]["sql"] + " WHERE assignment = 'offered'"),
            ("experiments", "start", "2024-12-31"),
            ("experiments", "end", "2025-01-04"),
            ("experiments", "observation_end", "2025-01-11"),
            ("experiments", "day_boundary", "UTC-05:00"),
            ("experiments", "control_group", "withheld"),
            ("design", "offered_group", "sent"),
            ("design", "groups_per_policy", {"all": {"base": 1, "middle": 2, "upper": 3}}),
        ):
            catalog = yaml.safe_load(original)
            entry = (
                catalog["experiments"][0]["design"] if section == "design" else catalog[section][0]
            )
            entry[key] = replacement
            path.write_text(yaml.safe_dump(catalog))
            rebound, logs = from_definitions(path, con, experiment_name="pilot")
            with logs, pytest.raises(CodedError) as changed:
                run(registered, logs)
            assert changed.value.code == "kernel.run.execution_unavailable", (section, key)
            assert registered.registration != rebound.registration, (section, key)
    finally:
        con.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    "late_person",
    [None, "store-1/2", "store-1/0"],
    ids=["empty_feed", "late_buyer", "late_offered"],
)
def test_full_window_requires_both_cutoff_and_certified_purchase_feed(con, tmp_path, late_person):
    import pyarrow as pa

    from increment.artifacts import SourceSnapshotEvidence
    from increment.store import ParquetResultsStore

    design, frames = three_policy_trial()
    required = WINDOW_END
    if late_person is not None:
        anchors = frames["assignments"]
        # Shift only this person's draw, leaving the original purchase stream untouched.
        assigned_at = anchors["assigned_at"].to_pylist()
        assigned_at[anchors["person_id"].to_pylist().index(late_person)] = ANCHOR + timedelta(
            hours=12
        )
        frames["assignments"] = anchors.set_column(
            anchors.schema.get_field_index("assigned_at"),
            "assigned_at",
            pa.array(assigned_at, type=anchors["assigned_at"].type),
        )
        required += timedelta(days=1)
    else:
        frames["purchases"] = frames["purchases"].slice(0, 0)
    store = ParquetResultsStore(tmp_path / "saved")
    short = required - timedelta(microseconds=1)
    receipts = (
        (complete_receipt(),)
        if late_person is not None
        else (
            None,
            SourceSnapshotEvidence(required, {"another_feed": required}),
            SourceSnapshotEvidence(required, {"purchases": None}),
            SourceSnapshotEvidence(required, {"purchases": short}),
            SourceSnapshotEvidence(short, {"purchases": required}),
        )
    )
    with launched_study(con, tmp_path, design, frames) as (path, bound_con, saved):
        registered = Experiment.from_json(saved)
        for receipt in receipts:
            _, reader = from_definitions(
                path, bound_con, experiment_name="pilot", source_snapshot_evidence=receipt
            )
            with reader, pytest.raises(CodedError) as incomplete:
                run(registered, reader, run_id="final", results_store=store)
            assert incomplete.value.code == "spillover.window.incomplete"
            assert incomplete.value.context["metric"] == "bought"
            assert incomplete.value.context["window_days"] == 7
            with pytest.raises(CodedError) as absent:
                store.open_run("pilot", registered.analysis_spec_hash, "final")
            assert absent.value.code == "store.run.not_found"
        _, reader = from_definitions(
            path,
            bound_con,
            experiment_name="pilot",
            source_snapshot_evidence=SourceSnapshotEvidence(required, {"purchases": required}),
        )
        with reader:
            rows = spillover_rows(run(registered, reader, run_id="final", results_store=store))
    expected = (0.030158730158730190, 0.480952380952380909) if late_person else (0.0, 0.0)
    assert [row.estimate for row in rows] == pytest.approx(expected, rel=0, abs=1e-14)
