"""Run the fixed spillover planning examples. Run as `python -m calibration.spillover`."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from increment.spillover import SpilloverDesign, SpilloverGroup, plan_spillover


def main() -> None:
    """Compare signed effect targets on fixed synthetic rosters; emit every result."""
    for group_count in (240, 500, 1000):
        groups = tuple(
            SpilloverGroup(f"g{i:04d}", tuple(f"{i}:{j}" for j in range(20)), 5, 10)
            for i in range(group_count)
        )
        design = SpilloverDesign(groups, {"all": group_count // 2}, "ordered", "one week")
        rng = np.random.default_rng(501)
        high_ids = set(rng.choice(group_count, size=group_count // 2, replace=False))
        high = {g.group_id: i in high_ids for i, g in enumerate(groups)}
        treated: dict[str, bool] = {}
        outcomes: dict[str, int] = {}
        for g in groups:
            quota = g.high_treated if high[g.group_id] else g.low_treated
            offered = set(rng.choice(20, size=quota, replace=False))
            treated.update((u, j in offered) for j, u in enumerate(g.members))
            outcomes.update((u, int(j < 5)) for j, u in enumerate(g.members) if j not in offered)
        for change in (0.10, 0.01):
            plan = plan_spillover(design, baseline_rate=0.25, meaningful_effect=change)
            readout = plan.analyze(group_high=high, treated=treated, outcomes=outcomes)
            print(
                json.dumps(
                    {
                        "groups": group_count,
                        "target": change,
                        "evidence": asdict(plan.evidence),
                        "readout": asdict(readout),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
