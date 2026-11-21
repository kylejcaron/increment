"""An actual two-stage trial keeps its point estimate when it cannot qualify an interval."""

from __future__ import annotations

from dataclasses import replace
from math import sqrt
from statistics import NormalDist
from typing import Any, cast

import pandas as pd
import pytest
import yaml

from increment import from_definitions, run
from increment.errors import CodedError
from increment.spillover import SpilloverComparison, SpilloverDesign, from_frame


def _trial(groups=4, size=4, low=1, high=2, baseline=None):
    names = [f"g{i:03}" for i in range(groups)]
    policies = ["high" if i < groups // 2 else "low" for i in range(groups)]
    roster = pd.DataFrame(
        [(f"{i}:{j}", g, "all") for i, g in enumerate(names) for j in range(size)],
        columns=["person_id", "group_id", "stratum"],
    )
    counts = pd.DataFrame(
        [(g, p, k) for g in names for p, k in (("low", low), ("high", high))],
        columns=["group_id", "policy", "offers"],
    )
    offered = pd.DataFrame(
        [
            (f"{i}:{j}", j < (high if p == "high" else low))
            for i, p in enumerate(policies)
            for j in range(size)
        ],
        columns=["person_id", "offered"],
    )
    design = SpilloverDesign(
        groups_per_policy={"all": {"low": groups // 2, "high": groups // 2}},
        comparisons=(
            SpilloverComparison(
                policy="high", reference="low", meaningful_effect=0.10, baseline_rate=baseline
            ),
        ),
        outcome="ordered",
        horizon="one week",
    )
    return roster, counts, design, pd.DataFrame({"group_id": names, "policy": policies}), offered


def test_unassessed_trial_keeps_point_and_distinguishes_observed_scarcity():
    roster, counts, design, policies, offered = _trial()
    events = offered.loc[~offered.offered, ["person_id"]].copy()
    events["outcome"] = events.person_id.isin({"0:2", "1:2", "1:3", "3:1"})
    experiment, reader = from_frame(
        roster,
        offer_counts=counts,
        design=design,
        group_policy=policies,
        offered=offered,
        outcomes=events,
    )
    with reader:
        (result,) = run(experiment, reader).comparisons
    assert (result.policy, result.reference, result.family_size) == ("high", "low", 1)
    assert result.estimate == pytest.approx(7 / 12)
    assert result.lower is None and result.upper is None
    assert result.qualification_reason == experiment.planning[0].reason == "missing_baseline_rate"
    assert result.observed_reason is None
    assert (result.outcome, result.horizon) == ("ordered", "one week")
    experiment, reader = from_frame(
        roster,
        offer_counts=counts,
        design=design,
        group_policy=policies,
        offered=offered,
        outcomes=events.assign(outcome=0),
        expected_registration=experiment.registration_id,
    )
    with reader:
        (zero,) = run(experiment, reader).comparisons
    assert zero.estimate == 0 and zero.observed_reason == "no_outcome_variation"
    assert zero.qualification_reason == result.qualification_reason


@pytest.mark.slow
def test_underpowered_trial_reports_qualified_interval_unless_observed_outcomes_are_constant():
    roster, counts, design, policies, offered = _trial(240, 20, 5, 10, 0.25)
    events = offered.loc[~offered.offered, ["person_id"]].copy()
    group_index = events.person_id.str.split(":").str[0].astype(int)
    events["outcome"] = (group_index < 45) | group_index.between(120, 134)
    experiment, reader = from_frame(
        roster,
        offer_counts=counts,
        design=design,
        group_policy=policies,
        offered=offered,
        outcomes=events,
    )
    assert experiment.planning[0].status == "underpowered"
    with reader:
        (result,) = run(experiment, reader).comparisons
        events["outcome"] = False
        (zero,) = run(experiment, reader).comparisons
    # Office rates are 0 or 1. The two arms have 45/120 and 15/120 successes.
    margin = NormalDist().inv_cdf(0.975) * sqrt((0.375 * 0.625 + 0.125 * 0.875) / 119)
    assert result.estimate == pytest.approx(0.25)
    assert (result.lower, result.upper) == pytest.approx((0.25 - margin, 0.25 + margin))
    assert result.unavailable_reason is None
    assert zero.estimate == 0 and zero.lower is None and zero.upper is None
    assert zero.observed_reason == "no_outcome_variation" and zero.qualification_reason is None


def test_registration_reopen_and_reader_lifecycle_before_observation():
    roster, counts, design, _, _ = _trial()
    experiment, reader = from_frame(roster, offer_counts=counts, design=design)
    with reader:
        reopened, other = from_frame(
            roster.iloc[::-1],
            offer_counts=counts.iloc[::-1],
            design=design,
            expected_registration=experiment.registration_id,
        )
        assert reopened.registration_id == experiment.registration_id
        with other:
            different, wrong = from_frame(
                roster, offer_counts=counts, design=design, experiment_id="other"
            )
            with wrong, pytest.raises(CodedError) as mismatch:
                run(different, reader)
            assert mismatch.value.code == "kernel.run.execution_unavailable"
    assert reader.closed
    with pytest.raises(CodedError) as closed:
        run(experiment, reader)
    assert closed.value.code == "kernel.run.reader_closed"
    swapped = roster.copy()
    swapped.loc[[0, 4], "group_id"] = swapped.loc[[4, 0], "group_id"].to_list()
    with pytest.raises(CodedError) as stale:
        from_frame(
            swapped,
            offer_counts=counts,
            design=design,
            expected_registration=experiment.registration_id,
        )
    assert stale.value.code == "spillover.roster.changed"


def test_registration_binds_comparison_and_group_specific_offer_counts():
    roster, counts, design, _, _ = _trial()
    original, reader = from_frame(roster, offer_counts=counts, design=design)
    reverse = SpilloverComparison(policy="low", reference="high", meaningful_effect=-0.10)
    with reader:
        for declaration, schedule in (
            (replace(design, outcome="clicked"), counts),
            (replace(design, comparisons=(reverse,)), counts),
            (design, counts.assign(offers=counts.offers.where(counts.index != 0, 0))),
        ):
            with pytest.raises(CodedError) as stale:
                from_frame(
                    roster,
                    offer_counts=schedule,
                    design=declaration,
                    expected_registration=original.registration_id,
                )
            assert stale.value.code == "spillover.roster.changed"


def test_unsupported_run_options_refuse_before_missing_logs():
    roster, counts, design, _, _ = _trial()
    pair = from_frame(roster, offer_counts=counts, design=design)
    with pair[1]:
        for options in (
            {"views": []},
            {"views": None},
            {"execution_context": object()},
            {"results_store": object()},
            {"decision_log": object()},
            {"run_id": "r"},
            {"capture": object()},
            {"prior": None},
            {"coherence": "live"},
            {"decision_method": object()},
            {"sensitivity_methods": []},
            {"metrics": []},
            {"estimands": []},
            {"value_scale": {}},
            {"exploratory_metrics": []},
            {"population": "assigned"},
            {"typo": 0},
        ):
            with pytest.raises(CodedError) as refused:
                cast(Any, run)(*pair, **options)
            assert refused.value.code == "kernel.request.invalid", options


def test_definitions_refuse_incompatible_options_and_offline_before_loading(tmp_path):
    path = tmp_path / "absent.yaml"
    for options in (
        {"backend": "duckdb"},
        {"store": "none"},
        {"store": "auto"},
        {"catalog_path": path},
        {"source_snapshot_evidence": object()},
        {"on_mixed_assignment": "exclude"},
        {"typo": 0},
    ):
        with pytest.raises(CodedError) as refusal:
            cast(Any, from_definitions)(path, object(), family="spillover", **options)
        assert refusal.value.code == "kernel.request.invalid", options
    with pytest.raises(CodedError) as offline:
        cast(Any, from_definitions)(path, family="spillover")
    assert offline.value.code == "kernel.request.invalid"


def test_spillover_yaml_is_strict_and_selects_named_experiment_before_reads(tmp_path):
    _, _, design, _, _ = _trial()
    relations = {
        key: key for key in ("roster", "offer_counts", "group_policy", "offered", "outcomes")
    }
    base: dict[str, Any] = {"name": "a", "relations": relations, "design": design.declaration()}
    path = tmp_path / "study.yaml"
    for entry, code in (
        ({**base, "members": ["private person"]}, "spillover.design.invalid"),
        (
            {**base, "relations": {**relations, "outcomes": "select * from events"}},
            "spillover.design.invalid",
        ),
        ({**base, "design": {**base["design"], "priors": []}}, "spillover.design.invalid"),
        (base, "definition.duplicates"),
        ({**base, "name": "b"}, "kernel.request.invalid"),
    ):
        path.write_text(yaml.safe_dump({"spillovers": [base, entry]}))
        with pytest.raises(CodedError) as refusal:
            cast(Any, from_definitions)(path, object(), family="spillover")
        assert refusal.value.code == code
    with pytest.raises(CodedError) as unknown:
        cast(Any, from_definitions)(path, object(), family="spillover", experiment_name="not-there")
    assert unknown.value.code == "facade.analysis.unknown_experiment"
