"""Assignment and outcome integrity for a two-stage offer experiment."""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from increment._frame_spillover import FrameSpilloverSource
from increment._spillover_design import (
    GroupProfile,
    PolicyCell,
    SpilloverComparison,
    SpilloverDesign,
)
from increment.errors import InvalidRequestError


def _comparison(policy="half", reference="zero"):
    return SpilloverComparison(policy=policy, reference=reference, meaningful_effect=-0.1)


def _design(**changes):
    options: dict[str, Any] = {
        "groups_per_policy": {"all": {"zero": 2, "half": 2, "full": 2}},
        "comparisons": [_comparison()],
        "outcome": "visited",
        "horizon": "seven days",
    }
    return SpilloverDesign(**(options | changes))


def _frames():
    # Sorted integer group keys differ from sorting their string representations.
    groups = (10, 2, 12, 4, 14, 6)
    policies = ("half", "zero", "full", "half", "zero", "full")
    sizes = (4, 3, 2, 6, 5, 4)
    events = (1, 0, 0, 3, 2, 0)
    roster, counts, offered, outcomes = [], [], [], []
    for g, n, p, e in zip(groups, sizes, policies, events, strict=True):
        quotas = {"zero": 0, "half": n // 2, "full": n}
        counts.extend((g, name, k) for name, k in quotas.items())
        for j in range(n):
            person = f"person/{g}/{j}"
            roster.append((person, g, "all"))
            offered.append((person, j < quotas[p]))
            if j >= quotas[p]:
                outcomes.append((person, int(j - quotas[p] < e)))
    return {
        "roster": pd.DataFrame(roster, columns=["person_id", "group_id", "stratum"]),
        "offer_counts": pd.DataFrame(counts, columns=["group_id", "policy", "offers"]),
        "group_policy": pd.DataFrame({"group_id": groups, "policy": policies}),
        "offered": pd.DataFrame(offered, columns=["person_id", "offered"]),
        "outcomes": pd.DataFrame(outcomes, columns=["person_id", "outcome"]),
    }


def test_freezes_declared_order_and_snapshots_all_nested_values():
    quotas = {"z": {"b": np.int64(1), "a": 1, "c": 1}, "a": {"c": 1, "b": 1, "a": 1}}
    pairs = [_comparison("c", "b"), _comparison("a", "b")]
    design = _design(groups_per_policy=quotas, comparisons=pairs)
    before = design.declaration()
    quotas["z"].clear()
    quotas.clear()
    pairs.clear()
    assert design.policies == ("a", "b", "c")
    assert design.comparisons == (_comparison("c", "b"), _comparison("a", "b"))
    assert design.declaration() == before
    assert design.groups_per_policy["z"]["b"] == 1
    with pytest.raises(TypeError):
        cast(Any, design.groups_per_policy["z"])["b"] = 2
    with pytest.raises(FrozenInstanceError):
        cast(Any, design.comparisons[0]).reference = "c"


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"horizon": " "}, "invalid"),
        ({"outcome": None}, "invalid"),
        ({"groups_per_policy": {}}, "quota"),
        ({"groups_per_policy": {"bad": {"only": 5}}}, "quota"),
        ({"groups_per_policy": {"x": {"a": 1, "b": 1}, "y": {"b": 1, "c": 1}}}, "quota"),
        ({"groups_per_policy": {"x": {"a": True, "b": 1}}}, "quota"),
        ({"groups_per_policy": {"x": {"a": 0, "b": 1}}}, "quota"),
        ({"comparisons": []}, "invalid"),
        (
            {"comparisons": [{"policy": "half", "reference": "zero", "meaningful_effect": 0.1}]},
            "invalid",
        ),
        ({"comparisons": [_comparison("new", "half")]}, "invalid"),
        ({"comparisons": [_comparison(), _comparison()]}, "duplicate"),
        ({"comparisons": [_comparison(), _comparison("zero", "half")]}, "duplicate"),
    ],
)
def test_rejects_unidentified_laws_and_ambiguous_comparisons(changes, code):
    with pytest.raises(InvalidRequestError) as exc:
        _design(**changes)
    assert exc.value.code == f"spillover.design.{code}"


@pytest.mark.parametrize(("p", "r"), [("same", "same"), ("", "other"), ("p", None)])
def test_comparison_needs_distinct_named_policies(p, r):
    with pytest.raises(InvalidRequestError) as exc:
        _comparison(p, r)
    assert exc.value.code == "spillover.design.invalid"


def test_frame_summary_uses_equal_group_weights_and_only_unoffered_outcomes():
    frames = _frames()
    source = FrameSpilloverSource(**frames)
    population = source.freeze(_design())
    assert [g.group_id for g in population.groups] == [2, 4, 6, 10, 12, 14]
    assert population.groups[0] == GroupProfile(2, "all", 3, (3, 1, 0))
    # Half: group totals 2 and 6; zero: group totals 0 and 2.
    expected = (
        PolicyCell("all", "half", 2, 4.0, 8.0, 5, 4),
        PolicyCell("all", "zero", 2, 1.0, 2.0, 8, 2),
    )
    assert source.observe(population) == expected
    offered_only = frames["offered"].loc[lambda t: t.offered, ["person_id"]].copy()
    offered_only["outcome"] = pd.array([1, *([None] * (len(offered_only) - 1))], dtype="Int64")
    frames["outcomes"] = pd.concat([offered_only, frames["outcomes"]], ignore_index=True)
    shuffled = FrameSpilloverSource(**{k: v.iloc[::-1] for k, v in frames.items()})
    assert shuffled.freeze(_design()) == population
    assert shuffled.observe(population) == expected


@pytest.mark.parametrize("field", ["roster", "offer_counts"])
def test_freeze_requires_projected_frames_not_mappings(field):
    frames = _frames()
    with pytest.raises(InvalidRequestError) as exc:
        inputs: dict[str, Any] = frames | {field: frames[field].to_dict()}
        FrameSpilloverSource(**inputs).freeze(_design())
    assert exc.value.code == "spillover.design.invalid"
    assert exc.value.context["field"] == field


@pytest.mark.parametrize(
    ("role", "change", "code"),
    [
        ("roster", "duplicate_person", "duplicate"),
        ("roster", "two_strata", "duplicate"),
        ("roster", "blank_id", "invalid"),
        ("roster", "mixed_ids", "invalid"),
        ("offer_counts", "wrong_domain", "invalid"),
        ("offer_counts", "duplicate_pair", "duplicate"),
        ("offer_counts", "missing_quota", "quota"),
        ("offer_counts", "unknown_pair", "quota"),
        ("offer_counts", "fraction", "quota"),
        ("offer_counts", "boolean", "quota"),
        ("offer_counts", "duration", "quota"),
        ("offer_counts", "saturated_comparison", "quota"),
    ],
)
def test_prelaunch_inputs_require_unique_ownership_and_feasible_counts(role, change, code):
    frames = _frames()
    df = frames[role].astype(object)
    frames[role] = df
    if change == "duplicate_person":
        frames[role] = pd.concat([df, df.iloc[:1].assign(group_id=99, stratum="other")])
    elif change == "duplicate_pair":
        frames[role] = pd.concat([df, df.iloc[:1]])
    elif change == "two_strata":
        df.at[0, "stratum"] = "other"
    elif change in ("blank_id", "mixed_ids", "wrong_domain"):
        df.at[0, "group_id"] = " " if change == "blank_id" else "10"
    elif change == "missing_quota":
        frames[role] = df.iloc[1:]
    elif change == "unknown_pair":
        df.at[0, "policy"] = "undeclared"
    else:
        df.at[0, "offers"] = {
            "fraction": 1.0,
            "boolean": True,
            "duration": np.timedelta64(1, "ns"),
            "saturated_comparison": 4,
        }[change]
    with pytest.raises(InvalidRequestError) as exc:
        FrameSpilloverSource(**frames).freeze(_design())
    assert exc.value.code == f"spillover.design.{code}"
    assert exc.value.context["field"] == role


@pytest.mark.parametrize("change", ["membership", "counts", "id_domain"])
def test_readout_rechecks_frozen_membership_and_every_quota(change):
    frames = _frames()
    population = FrameSpilloverSource(**frames).freeze(_design())
    if change == "membership":
        df = frames["roster"]
        df.loc[[0, 4], "group_id"] = [2, 10]
    elif change == "counts":
        # Unassigned potential quota: no realized offer count would reveal this change.
        frames["offer_counts"].at[0, "offers"] = 1
    else:
        for role in ("roster", "offer_counts", "group_policy"):
            frames[role]["group_id"] = frames[role].group_id.astype(str)
    with pytest.raises(InvalidRequestError) as exc:
        FrameSpilloverSource(**frames).observe(population)
    assert exc.value.code == "spillover.roster.changed"
    assert set(exc.value.context) == {"field", "requirement"}


@pytest.mark.parametrize("field", ["group_policy", "offered", "outcomes"])
@pytest.mark.parametrize("mismatch", ["missing", "foreign", "duplicate"])
def test_log_keys_never_drop_foreign_or_duplicate_rows_or_impute_missing(field, mismatch):
    frames = _frames()
    log = frames[field]
    if mismatch == "missing":
        frames[field] = log.iloc[1:]
    elif mismatch == "duplicate":
        frames[field] = pd.concat([log, log.iloc[:1]])
    else:
        log.iat[0, 0] = -1 if field == "group_policy" else "foreign"
    source = FrameSpilloverSource(**frames)
    with pytest.raises(InvalidRequestError) as exc:
        source.observe(source.freeze(_design()))
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == field


@pytest.mark.parametrize(
    ("role", "column", "value", "code"),
    [
        ("group_policy", "group_id", "10", "keys"),
        ("group_policy", "policy", 1, "keys"),
        ("offered", "offered", 2, "binary"),
        ("offered", "offered", "1", "binary"),
        ("offered", "offered", None, "binary"),
        ("outcomes", "outcome", 0.0, "binary"),
        ("outcomes", "outcome", None, "keys"),
    ],
)
def test_logs_require_binary_measurements_and_matching_key_domains(role, column, value, code):
    frames = _frames()
    frames[role] = frames[role].astype(object)
    frames[role].at[0, column] = value
    source = FrameSpilloverSource(**frames)
    with pytest.raises(InvalidRequestError) as exc:
        source.observe(source.freeze(_design()))
    assert exc.value.code == f"spillover.log.{code}"
    assert exc.value.context["field"] == role


def test_optional_offered_measurement_cannot_be_nonbinary_even_in_uncompared_group():
    frames = _frames()
    frames["outcomes"].loc[len(frames["outcomes"])] = ["person/12/0", 2]
    source = FrameSpilloverSource(**frames)
    with pytest.raises(InvalidRequestError) as exc:
        source.observe(source.freeze(_design()))
    assert exc.value.code == "spillover.log.binary"


@pytest.mark.parametrize("field", ["group_policy", "offered"])
def test_both_realized_quotas_are_local_not_just_trial_totals(field):
    frames = _frames()
    roster = frames["roster"]
    roster["stratum"] = roster.group_id.map(lambda g: "west" if g in {10, 2, 12} else "east")
    design = _design(
        groups_per_policy={s: dict.fromkeys(_design().policies, 1) for s in ("east", "west")}
    )
    if field == "group_policy":
        frames[field].loc[[0, 4], "policy"] = ["zero", "half"]
    else:
        df = frames[field]
        df.loc[df.person_id == "person/10/0", "offered"] = False
        df.loc[df.person_id == "person/2/0", "offered"] = True
    source = FrameSpilloverSource(**frames)
    population = source.freeze(design)
    assert [(g.stratum, g.group_id) for g in population.groups] == [
        ("east", 4),
        ("east", 6),
        ("east", 14),
        ("west", 2),
        ("west", 10),
        ("west", 12),
    ]
    with pytest.raises(InvalidRequestError) as exc:
        source.observe(population)
    assert exc.value.code == "spillover.log.quota"
    assert exc.value.context["field"] == field


def test_stratum_totals_must_account_for_every_group():
    frames = _frames()
    design = replace(_design(), groups_per_policy={"all": {"zero": 2, "half": 2, "full": 1}})
    with pytest.raises(InvalidRequestError) as exc:
        FrameSpilloverSource(**frames).freeze(design)
    assert exc.value.code == "spillover.design.quota"
    assert exc.value.context["field"] == "groups_per_policy"


def test_planning_needs_no_observed_frames_but_readout_requires_them():
    frames = _frames()
    source = FrameSpilloverSource(roster=frames["roster"], offer_counts=frames["offer_counts"])
    population = source.freeze(_design())
    with pytest.raises(InvalidRequestError) as exc:
        source.observe(population)
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == "group_policy"
