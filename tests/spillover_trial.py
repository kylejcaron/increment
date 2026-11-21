"""A three-policy experiment with unequal group sizes and independently calculated RCT2 answers."""

from __future__ import annotations

from contextlib import ExitStack
from uuid import uuid4

import pyarrow as pa
import pytest
import yaml

from increment import from_definitions, run

POLICIES = ("base", "middle", "upper")


def three_policy_trial():
    sizes = (6, 8, 5, 7, 9, 10)
    events = (0, 2, 2, 3, 1, 4)
    groups = [f"store-{j}" for j in range(len(sizes))]
    rosters, quotas, policies, offers, outcomes = [], [], [], [], []
    for j, (group, size, event_count) in enumerate(zip(groups, sizes, events, strict=True)):
        policy = j % 3
        policies.append({"group_id": group, "policy": POLICIES[policy]})
        for count, label in enumerate(POLICIES, start=1):
            quotas.append({"group_id": group, "policy": label, "offers": count})
        for i in range(size):
            person = f"{group}/{i}"
            offered = i < policy + 1
            rosters.append({"person_id": person, "group_id": group, "stratum": "all"})
            offers.append({"person_id": person, "offered": offered})
            if not offered:
                outcomes.append({"person_id": person, "outcome": i < policy + 1 + event_count})
    design = {
        "outcome": "bought",
        "horizon": "within seven days",
        "groups_per_policy": {"all": dict.fromkeys(POLICIES, 2)},
        "comparisons": [
            {"policy": label, "reference": "base", "meaningful_effect": 0.05}
            for label in POLICIES[1:]
        ],
    }
    relations = {
        "roster": pa.Table.from_pylist(rosters),
        "offer_counts": pa.Table.from_pylist(quotas),
        "group_policy": pa.Table.from_pylist(policies),
        "offered": pa.Table.from_pylist(offers),
        "outcomes": pa.Table.from_pylist(outcomes),
    }
    return design, relations


def frame_design(fields):
    from increment.spillover import SpilloverComparison, SpilloverDesign

    return SpilloverDesign(
        groups_per_policy=fields["groups_per_policy"],
        comparisons=tuple(SpilloverComparison(**c) for c in fields["comparisons"]),
        outcome=fields["outcome"],
        horizon=fields["horizon"],
    )


def write_study(path, design, names):
    path.write_text(
        yaml.safe_dump({"spillovers": [{"name": "pilot", "design": design, "relations": names}]}),
        encoding="utf-8",
    )


def warehouse_run(con, tmp_path, design, frames, *, reader_con=None):
    """Open before assignment logs exist; clean up even when the readout refuses."""
    tag = uuid4().hex[:12]
    names = {role: f"spill_{tag}_{role}" for role in frames}
    path = tmp_path / "spillover.yaml"
    declared = names
    if reader_con is not None:
        namespace = f"{con.current_catalog}.{con.current_database}"
        declared = {role: f"{namespace}.{name}" for role, name in names.items()}
    write_study(path, design, declared)
    with ExitStack() as cleanup:

        def load(role):
            con.create_table(names[role], obj=frames[role])
            cleanup.callback(con.drop_table, names[role], force=True)

        load("roster")
        load("offer_counts")
        experiment, reader = from_definitions(
            path, con if reader_con is None else reader_con, family="spillover"
        )
        cleanup.enter_context(reader)
        for role in ("group_policy", "offered", "outcomes"):
            load(role)
        return run(experiment, reader)


def run_three_policy_probe(con, tmp_path, *, reader_con=None):
    result = warehouse_run(con, tmp_path, *three_policy_trial(), reader_con=reader_con)
    assert [(row.policy, row.reference) for row in result.comparisons] == [
        ("middle", "base"),
        ("upper", "base"),
    ]
    # Pinned CalAPO on the full 45-person roster; other arms never become controls.
    assert [row.estimate for row in result.comparisons] == pytest.approx(
        (0.030158730158730190, 0.480952380952380909), abs=1e-14, rel=0
    )
    for row in result.comparisons:
        assert row.lower is row.upper is row.observed_reason is None
        assert row.qualification_reason == "missing_baseline_rate"
    return result
