"""Reproduce the two- and three-policy spillover planning examples."""

from __future__ import annotations

import argparse
import json
from time import monotonic

import numpy as np
import pyarrow as pa

from increment import run
from increment.spillover import SpilloverComparison, SpilloverDesign, from_frame


def offer_trial(level_counts, quotas, changes):
    """Generate fixed-size groups with a recorded random policy and person-level offer draw."""
    labels = tuple(level_counts)
    size = 20
    count = sum(level_counts.values())
    rng = np.random.default_rng(501)
    groups = [f"g{i:04d}" for i in range(count)]
    assignments = rng.permutation(np.repeat(labels, list(level_counts.values()))).tolist()
    # Only even-numbered groups have an event under the second policy; the
    # potential outcomes are fixed before either draw.
    person_ids, group_ids, offers, outcome_ids, events = [], [], [], [], []
    for j, (group, policy) in enumerate(zip(groups, assignments, strict=True)):
        selected = set(rng.choice(size, size=quotas[policy], replace=False))
        event = policy == labels[1] and j % 2 == 0
        for i in range(size):
            person = f"{group}/{i}"
            person_ids.append(person)
            group_ids.append(group)
            offers.append(i in selected)
            if i not in selected:
                outcome_ids.append(person)
                events.append(event)
    frames = {
        "roster": pa.table(
            {"group_id": group_ids, "person_id": person_ids, "stratum": ["all"] * len(person_ids)}
        ),
        "offer_counts": pa.table(
            {
                "group_id": [g for g in groups for _ in labels],
                "policy": list(labels) * count,
                "offers": [quotas[label] for label in labels] * count,
            }
        ),
        "group_policy": pa.table({"group_id": groups, "policy": assignments}),
        "offered": pa.table({"person_id": person_ids, "offered": offers}),
        "outcomes": pa.table({"person_id": outcome_ids, "outcome": events}),
    }
    design = SpilloverDesign(
        groups_per_policy={"all": level_counts},
        comparisons=tuple(
            SpilloverComparison(
                policy=p, reference=labels[0], meaningful_effect=effect, baseline_rate=0.25
            )
            for p, effect in zip(labels[1:], changes, strict=True)
        ),
        outcome="ordered",
        horizon="one week",
    )
    return design, frames


def three_levels():
    return offer_trial(
        {"base": 400, "middle": 300, "upper": 300},
        {"base": 0, "middle": 5, "upper": 10},
        (0.10, 0.10),
    )


def _report(design, frames):
    start = monotonic()
    experiment, reader = from_frame(**frames, design=design)
    elapsed = monotonic() - start
    with reader:
        result = run(experiment, reader)
    for planning, row in zip(experiment.planning, result.comparisons, strict=True):
        scenarios = planning.evidence.scenarios
        print(
            json.dumps(
                {
                    "groups": sum(design.groups_per_policy["all"].values()),
                    "comparison": [row.policy, row.reference],
                    "target": next(
                        c.meaningful_effect for c in design.comparisons if c.policy == row.policy
                    ),
                    "planning_seconds_for_family": elapsed,
                    "planning": planning.status,
                    "reason": planning.reason,
                    "attempts_per_scenario": None if not scenarios else scenarios[0].attempts,
                    "worst_miss_upper": max((s.miss_upper for s in scenarios), default=None),
                    "estimate": row.estimate,
                    "interval": [row.lower, row.upper],
                    "observed_reason": row.observed_reason,
                },
                allow_nan=False,
            ),
            flush=True,
        )


def main() -> None:
    """Run the three-policy study, or repeat the original two-policy launch comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--two-policy", action="store_true", help="repeat the six original launch plans"
    )
    if parser.parse_args().two_policy:
        for n in (240, 500, 1000):
            for change in (0.10, 0.01):
                _report(
                    *offer_trial(
                        {"base": n // 2, "upper": n // 2}, {"base": 5, "upper": 10}, (change,)
                    )
                )
    else:
        _report(*three_levels())


if __name__ == "__main__":
    main()
