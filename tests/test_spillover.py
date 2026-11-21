"""An actual two-stage trial keeps its point estimate when it cannot qualify an interval."""

from __future__ import annotations

import pytest

from increment.spillover import SpilloverDesign, SpilloverGroup, plan_spillover
from tests.capability_consumers_auxiliary import assert_support


def test_unassessed_trial_keeps_point_and_distinguishes_observed_scarcity():
    groups = tuple(
        SpilloverGroup(f"g{i}", tuple(f"{i}-{j}" for j in range(4)), 1, 2) for i in range(4)
    )
    design = SpilloverDesign(groups, {"all": 2}, outcome="ordered", horizon="one week")
    plan = plan_spillover(design, meaningful_effect=0.05)
    high = {g.group_id: i < 2 for i, g in enumerate(groups)}
    treated = {
        u: j < (2 if high[g.group_id] else 1) for g in groups for j, u in enumerate(g.members)
    }
    # High-group untreated rates 1/2 and 1; low-group rates 0 and 1/3.
    outcomes = {u: int(u in {"0-2", "1-2", "1-3", "3-1"}) for u in treated if not treated[u]}
    result = plan.analyze(group_high=high, treated=treated, outcomes=outcomes)
    assert_support("spillover_family_boundary", "untreated_contrast_supported", result)
    assert result.estimate == pytest.approx(7 / 12)
    assert result.lower is None and result.upper is None
    assert result.qualification_reason == plan.evidence.reason and result.qualification_reason
    assert result.observed_reason is None
    assert (result.outcome, result.horizon) == ("ordered", "one week")
    zero = plan.analyze(group_high=high, treated=treated, outcomes=dict.fromkeys(outcomes, 0))
    assert zero.estimate == 0 and zero.observed_reason == "no_outcome_variation"
    assert zero.qualification_reason == result.qualification_reason
