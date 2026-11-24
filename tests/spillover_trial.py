"""A three-policy experiment with unequal group sizes and independently calculated RCT2 answers."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pytest
import yaml

from increment import (
    AnalysisPlan,
    Experiment,
    MetricSpec,
    RunResult,
    Spillover,
    SpilloverComparison,
    from_definitions,
    from_frame,
    run,
)
from increment.estimation.spillover_results import SpilloverEffects

POLICIES = ("base", "middle", "upper")
ANCHOR = datetime(2025, 1, 1, 12, tzinfo=UTC)
WINDOW_END = datetime(2025, 1, 8, tzinfo=UTC)
WAREHOUSE_ROLES = ("roster", "offer_counts", "group_policy", "assignments", "purchases")


def three_policy_trial(*, alpha=0.01):
    sizes = (6, 8, 5, 7, 9, 10)
    events = (0, 2, 2, 3, 1, 4)
    groups = [f"store-{j}" for j in range(len(sizes))]
    people, quotas, policies, anchors, purchases = [], [], [], [], []
    for j, (group, size, event_count) in enumerate(zip(groups, sizes, events, strict=True)):
        policy = j % 3
        policies.append({"group_id": group, "policy": POLICIES[policy]})
        for count, label in enumerate(POLICIES, start=1):
            quotas.append({"group_id": group, "policy": label, "offers": count})
        for i in range(size):
            person = f"{group}/{i}"
            offered = i < policy + 1
            assignment = "offered" if offered else "no_offer"
            bought = not offered and i < policy + 1 + event_count
            people.append(
                {
                    "person_id": person,
                    "group_id": group,
                    "stratum": "all",
                    "assignment": assignment,
                    "bought": None if offered else bought,
                }
            )
            anchors.append(
                {
                    "person_id": person,
                    "assignment": assignment,
                    "assigned_at": ANCHOR,
                    "experiment_id": "pilot",
                }
            )
            if bought:
                purchases.append((person, ANCHOR + timedelta(days=1), "web"))
    # Only the first of these customers already has a qualifying purchase.
    purchases.extend(
        [
            ("store-1/2", ANCHOR + timedelta(days=2), "web"),
            ("store-0/1", ANCHOR, "web"),
            ("store-0/2", WINDOW_END, "web"),
            ("store-0/3", ANCHOR + timedelta(days=1), "shop"),
        ]
    )
    design = {
        "alpha": alpha,
        "groups_per_policy": {"all": dict.fromkeys(POLICIES, 2)},
        "comparisons": [
            {"policy": label, "reference": "base", "meaningful_effect": 0.05}
            for label in POLICIES[1:]
        ],
    }
    summary = pa.Table.from_pylist(people)
    relations = {
        "summary": summary,
        "roster": summary.select(["person_id", "group_id", "stratum"]),
        "offer_counts": pa.Table.from_pylist(quotas),
        "group_policy": pa.Table.from_pylist(policies),
        "assignments": pa.Table.from_pylist(anchors),
        "purchases": pa.table(
            list(zip(*purchases, strict=True)), names=["person_id", "purchased_at", "channel"]
        ),
    }
    return design, relations


def frame_design(fields):
    return Spillover(
        control_group="no_offer",
        offered_group="offered",
        groups_per_policy=fields["groups_per_policy"],
        comparisons=tuple(SpilloverComparison(**c) for c in fields["comparisons"]),
    )


def bind_summary(
    frame,
    counts,
    policies,
    design,
    *,
    metric="bought",
    window_days=7,
    alpha=0.05,
    experiment_id="pilot",
):
    return from_frame(
        frame,
        shape="summary",
        experiment_id=experiment_id,
        unit="person_id",
        group="assignment",
        cluster="group_id",
        stratum="stratum",
        design=design,
        metrics=[MetricSpec(name=metric, type="conversion", window_days=window_days)],
        plan=AnalysisPlan(primary=metric, alpha=alpha),
        offer_counts=counts,
        group_policy=policies,
    )


def frame_trial(design, frames, *, planning=False):
    return bind_summary(
        frames["roster" if planning else "summary"],
        frames["offer_counts"],
        None if planning else frames["group_policy"],
        frame_design(design),
        alpha=design.get("alpha", 0.05),
    )


def paired_trial(strata=40):
    """Matched stores: reference mean 1/4, other-policy mean 3/4 under every draw."""
    names = [f"s{i:06d}" for i in range(strata)]
    design = {
        "groups_per_policy": {s: {"base": 1, "upper": 1} for s in names},
        "comparisons": [
            {
                "policy": "upper",
                "reference": "base",
                "meaningful_effect": 0.5,
                "baseline_rate": 0.25,
            }
        ],
    }
    rng = np.random.default_rng(519_406)
    people, counts, policies = [], [], []
    for h, s in enumerate(names):
        high = int(rng.integers(2))
        for j in range(2):
            group = 2 * h + j
            policy = "upper" if j == high else "base"
            offered = set(rng.choice(4, 2, replace=False)) if j == high else set()
            bought = h < (3 * strata // 4 if j == high else strata // 4)
            policies.append({"group_id": group, "policy": policy})
            counts.extend(
                {"group_id": group, "policy": label, "offers": n}
                for label, n in (("base", 0), ("upper", 2))
            )
            people.extend(
                {
                    "person_id": 4 * group + i,
                    "group_id": group,
                    "stratum": s,
                    "assignment": "offered" if i in offered else "no_offer",
                    "bought": None if i in offered else bought,
                }
                for i in range(4)
            )
    summary = pa.Table.from_pylist(people)
    return design, {
        "summary": summary,
        "roster": summary.select(["person_id", "group_id", "stratum"]),
        "offer_counts": pa.Table.from_pylist(counts),
        "group_policy": pa.Table.from_pylist(policies),
    }


def write_study(path, design, names, *, dialect=None):
    import sqlglot as sg

    def select(role, *columns):
        table = sg.to_table(names[role], dialect=dialect)
        return sg.select(*columns).from_(table).sql(dialect=dialect, identify=True)

    unit = design.get("unit", "person_id")

    catalog = {
        "experiments": [
            {
                "name": "pilot",
                "unit": unit,
                "control_group": "no_offer",
                "exposure": "coupon_assignment",
                "start": "2025-01-01",
                "end": "2025-01-03",
                "observation_end": "2025-01-10",
                "design": {
                    "mechanism": "spillover",
                    "offered_group": "offered",
                    "groups_per_policy": design["groups_per_policy"],
                    "comparisons": design["comparisons"],
                    **{key: design[key] for key in ("cluster", "stratum") if key in design},
                    **{role: names[role] for role in ("roster", "offer_counts", "group_policy")},
                },
                "plan": {"primary": "bought", "alpha": design.get("alpha", 0.05)},
            }
        ],
        "exposures": [
            {
                "name": "coupon_assignment",
                "sql": select(
                    "assignments",
                    f"{unit} AS unit_id",
                    "assignment AS group_id",
                    "assigned_at AS ts",
                    "experiment_id",
                ),
            }
        ],
        "fact_sources": [
            {
                "name": "purchases",
                "sql": select("purchases", "*", "'coupon_purchase' AS event"),
                "timestamp_column": "purchased_at",
                "entities": [unit],
                "facts": [{"name": "coupon_purchase"}],
                "properties": [{"name": "channel", "column": "channel"}],
            }
        ],
        "metrics": [
            {
                "name": "bought",
                "entity": unit,
                "type": "conversion",
                "window_days": 7,
                "fact": "coupon_purchase",
                "filters": [{"property": "channel", "op": "equals", "values": ["web"]}],
            }
        ],
    }
    if dialect is not None:
        catalog["dialect"] = dialect
    path.write_text(yaml.safe_dump(catalog), encoding="utf-8")


def complete_receipt():
    from increment.artifacts import SourceSnapshotEvidence

    return SourceSnapshotEvidence(
        observation_cutoff_ts=WINDOW_END, complete_through_by_feed={"purchases": WINDOW_END}
    )


@contextmanager
def launched_study(con, tmp_path, design, frames, *, reader_con=None):
    """Freeze the declaration before any assignment or purchase tables exist."""
    tag = uuid4().hex[:12]
    names = {role: f"spill_{tag}_{role}" for role in WAREHOUSE_ROLES}
    path = tmp_path / "spillover.yaml"
    declared = names
    if reader_con is not None:
        namespace = f"{con.current_catalog}.{con.current_database}"
        declared = {role: f"{namespace}.{name}" for role, name in names.items()}
    write_study(path, design, declared, dialect=con.name)
    bound_con = con if reader_con is None else reader_con
    with ExitStack() as cleanup:

        def load(role):
            con.create_table(names[role], obj=frames[role])
            cleanup.callback(con.drop_table, names[role], force=True)

        load("roster")
        load("offer_counts")
        experiment, planning_reader = from_definitions(path, bound_con, experiment_name="pilot")
        with planning_reader:
            registered_json = experiment.to_json()
        for role in ("group_policy", "assignments", "purchases"):
            load(role)
        yield path, bound_con, registered_json


def warehouse_run(con, tmp_path, design, frames, *, reader_con=None):
    with launched_study(con, tmp_path, design, frames, reader_con=reader_con) as (
        path,
        bound_con,
        registered_json,
    ):
        _, reader = from_definitions(
            path, bound_con, experiment_name="pilot", source_snapshot_evidence=complete_receipt()
        )
        with reader:
            return run(Experiment.from_json(registered_json), reader)


def spillover_rows(result: RunResult) -> SpilloverEffects:
    rows = result.readout()
    assert isinstance(rows, SpilloverEffects)
    return rows


def run_three_policy_probe(con, tmp_path, *, alpha=0.01, reader_con=None):
    result = warehouse_run(con, tmp_path, *three_policy_trial(alpha=alpha), reader_con=reader_con)
    rows = spillover_rows(result)
    assert [(row.policy, row.reference) for row in rows] == [
        ("middle", "base"),
        ("upper", "base"),
    ]
    # Pinned CalAPO on the full 45-person roster; other arms never become controls.
    assert [row.estimate for row in rows] == pytest.approx(
        (0.030158730158730190, 0.480952380952380909), abs=1e-14, rel=0
    )
    # Policy X values: (0, 7/20), (4/15, 9/70), (1/2, 4/7).
    # Invert two-point moments for X and 1-X at bets .95/.475 and threshold 6/alpha.
    # Scale policy bounds by 60/45 and clip before differencing.
    ranges = {
        0.01: ((-0.995417077848, 0.999884521847), (-0.978388379672, 0.999884521847)),
        0.05: ((-0.988590287657, 0.999422073323), (-0.950994224644, 0.999422073323)),
    }
    for row, expected in zip(rows, ranges[alpha], strict=True):
        assert (row.metric, row.window_days) == ("bought", 7)
        assert (row.lower, row.upper) == pytest.approx(expected, rel=0, abs=1e-10)
        assert (row.alpha, row.family_size, row.policy_count) == (alpha, 2, 3)
        assert row.method == "two-stage-fixed-bet-v3"
        assert row.direction == "inconclusive" and row.unavailable_reason is None
    return result
