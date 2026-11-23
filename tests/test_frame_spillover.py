"""Assignment and outcome integrity for a two-stage offer experiment."""

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from increment import (
    AnalysisPlan,
    Experiment,
    MetricSpec,
    Spillover,
    SpilloverComparison,
    from_frame,
    run,
)
from increment.errors import CodedError, InvalidRequestError
from increment.estimation.spillover_results import SpilloverEffect, SpilloverEffects

_METRIC = MetricSpec(name="visits_7d", value_column="visited", type="conversion", window_days=7)


def _comparison(policy="half", reference="zero"):
    return SpilloverComparison(policy=policy, reference=reference, meaningful_effect=-0.1)


def _design(**changes):
    options: dict[str, Any] = {
        "groups_per_policy": {"all": {"zero": 2, "half": 2, "full": 2}},
        "comparisons": [_comparison()],
        "control_group": "no offer",
        "offered_group": "coupon",
    }
    return Spillover(**(options | changes))


def _frames():
    # Sorted integer group keys differ from sorting their string representations.
    groups = (10, 2, 12, 4, 14, 6)
    policies = ("half", "zero", "full", "half", "zero", "full")
    sizes = (4, 3, 2, 6, 5, 4)
    events = (1, 0, 0, 3, 2, 0)
    members, counts = [], []
    for g, n, p, e in zip(groups, sizes, policies, events, strict=True):
        quotas = {"zero": 0, "half": n // 2, "full": n}
        counts.extend((g, name, k) for name, k in quotas.items())
        for j in range(n):
            person = f"person/{g}/{j}"
            assignment = "coupon" if j < quotas[p] else "no offer"
            event = None if j < quotas[p] else int(j - quotas[p] < e)
            members.append((person, g, "all", assignment, event))
    roster = pd.DataFrame(members, columns=["customer", "store", "region", "assignment", "visited"])
    return {
        "roster": roster,
        "offer_counts": pd.DataFrame(counts, columns=["store", "policy", "offers"]),
        "group_policy": pd.DataFrame({"store": groups, "policy": policies}),
    }


def _inputs(frames, design, stratum) -> dict[str, Any]:
    return {
        "unit": "customer",
        "group": "assignment",
        "cluster": "store",
        "stratum": stratum,
        "design": _design() if design is None else design,
        "offer_counts": frames["offer_counts"],
        "group_policy": frames.get("group_policy"),
    }


def _register(frames, design=None, *, stratum="region", alpha=0.05, metric=_METRIC):
    return from_frame(
        frames["roster"],
        shape="summary",
        metrics=[metric],
        plan=AnalysisPlan(primary=metric.name, alpha=alpha),
        **_inputs(frames, design, stratum),
    )


def _effect(experiment, reader) -> SpilloverEffect:
    effects = run(experiment, reader).readout()
    assert isinstance(effects, SpilloverEffects)
    (row,) = effects
    return row


def test_freezes_declared_order_and_snapshots_all_nested_values():
    quotas = {"z": {"b": np.int64(1), "a": 1, "c": 1}, "a": {"c": 1, "b": 1, "a": 1}}
    pairs = [_comparison("c", "b"), _comparison("a", "b")]
    design = _design(groups_per_policy=quotas, comparisons=pairs).law("visited", 7, 0.05)
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


def test_alpha_is_a_preassignment_decision_not_a_readout_option():
    frames = _frames()
    default, reader = _register(frames)
    with reader:
        strict, other = _register(frames, alpha=0.01)
        with other:
            assert strict.registration != default.registration
            assert _effect(strict, other).alpha == 0.01
            with pytest.raises(CodedError) as stale:
                run(default, other)
        assert stale.value.code == "kernel.run.execution_unavailable"
        assert stale.value.context["field"] == "design"
    for invalid in (0, 1, -0.1, True, np.bool_(False), float("nan"), float("inf"), "0.05"):
        with pytest.raises(InvalidRequestError) as error:
            _design().law("visited", 7, cast(Any, invalid))
        assert error.value.context["field"] == "alpha"


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
        replace(_design().law("visited", 7, 0.05), **changes)
    assert exc.value.code == f"spillover.design.{code}"


@pytest.mark.parametrize(("p", "r"), [("same", "same"), ("", "other"), ("p", None)])
def test_comparison_needs_distinct_named_policies(p, r):
    with pytest.raises(InvalidRequestError) as exc:
        _comparison(p, r)
    assert exc.value.code == "spillover.design.invalid"


@pytest.mark.parametrize("backend", ["pandas", "arrow", "polars"])
def test_frame_summary_uses_equal_group_weights_and_only_unoffered_outcomes(backend):
    frames = _frames()
    experiment, reader = _register(frames)
    assert experiment.analysis_context["planning_groups"][0] == ("all", 3, (3, 1, 0))
    with reader:
        row = _effect(experiment, reader)
    assert row.registration_hash == experiment.registration.registration_hash
    # Expand untreated events to group totals; two groups per arm represent all six.
    half = (4 * 1 / 2 + 6 * 3 / 3) / 2
    zero = (0 + 2) / 2
    assert (row.policy, row.reference) == ("half", "zero")
    assert row.estimate == (half - zero) * 6 / 24
    # Exact moments of the two compared policies: [2, 6] and [0, 2], B=6, N=24.
    assert (row.lower, row.upper) == pytest.approx((-0.964962853, 0.999070650), abs=1e-9)
    assert row.direction == "inconclusive" and row.unavailable_reason is None

    frames["roster"].at[0, "visited"] = -99
    shuffled: dict[str, Any] = {k: v.iloc[::-1] for k, v in frames.items()}
    if backend == "arrow":
        shuffled = {k: pa.Table.from_pandas(v, preserve_index=False) for k, v in shuffled.items()}
    elif backend == "polars":
        shuffled = {k: pl.from_pandas(v) for k, v in shuffled.items()}
    rebound, fresh = _register(shuffled)
    assert rebound.registration == experiment.registration
    with fresh:
        assert _effect(Experiment.from_json(experiment.to_json()), fresh) == row


@pytest.mark.parametrize("field", ["roster", "offer_counts"])
def test_freeze_requires_projected_frames_not_mappings(field):
    frames = _frames()
    with pytest.raises(InvalidRequestError) as exc:
        inputs: dict[str, Any] = frames | {field: frames[field].to_dict()}
        _register(inputs)
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
        frames[role] = pd.concat([df, df.iloc[:1].assign(store=99, region="other")])
    elif change == "duplicate_pair":
        frames[role] = pd.concat([df, df.iloc[:1]])
    elif change == "two_strata":
        df.at[0, "region"] = "other"
    elif change in ("blank_id", "mixed_ids", "wrong_domain"):
        df.at[0, "store"] = " " if change == "blank_id" else "10"
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
        _register(frames)
    assert exc.value.code == f"spillover.design.{code}"
    assert exc.value.context["field"] == role


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("membership", "spillover.roster.changed"),
        ("counts", "spillover.roster.changed"),
        ("id_domain", "spillover.roster.changed"),
        ("missing_person", "spillover.roster.changed"),
        ("foreign_person", "spillover.roster.changed"),
        ("duplicate_person", "spillover.design.duplicate"),
    ],
)
def test_readout_rechecks_frozen_membership_and_every_quota(change, code):
    frames = _frames()
    counts = frames["offer_counts"]
    # Even losing one person would leave every prospective quota feasible in store 10.
    counts.loc[(counts.store == 10) & (counts.policy == "full"), "offers"] = 3
    experiment, reader = _register(frames)
    if change == "membership":
        df = frames["roster"]
        df.loc[[0, 4], "store"] = [2, 10]
    elif change == "counts":
        # Unassigned potential quota: no realized offer count would reveal this change.
        frames["offer_counts"].at[0, "offers"] = 1
    elif change == "missing_person":
        frames["roster"].drop(index=0, inplace=True)
    elif change == "foreign_person":
        frames["roster"].at[0, "customer"] = "foreign customer"
    elif change == "duplicate_person":
        frames["roster"].at[0, "customer"] = frames["roster"].at[1, "customer"]
    else:
        for role in ("roster", "offer_counts", "group_policy"):
            frames[role]["store"] = frames[role].store.astype(str)
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == code
    assert set(exc.value.context) == {"field", "requirement"}
    assert exc.value.context["field"] == "roster"


@pytest.mark.parametrize("mismatch", ["missing", "foreign", "duplicate"])
def test_group_policy_log_never_drops_foreign_or_duplicate_rows_or_imputes_missing(mismatch):
    frames = _frames()
    log = frames["group_policy"]
    if mismatch == "missing":
        frames["group_policy"] = log.iloc[1:]
    elif mismatch == "duplicate":
        frames["group_policy"] = pd.concat([log, log.iloc[:1]])
    else:
        log.iat[0, 0] = -1
    experiment, reader = _register(frames)
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == "group_policy"


@pytest.mark.parametrize(
    ("role", "column", "value", "code", "field"),
    [
        ("group_policy", "store", "10", "keys", "group_policy"),
        ("group_policy", "policy", 1, "keys", "group_policy"),
        ("roster", "assignment", 2, "keys", "offered"),
        ("roster", "assignment", "1", "keys", "offered"),
        ("roster", "assignment", pd.NA, "keys", "offered"),
        ("roster", "visited", np.nextafter(1.0, 2.0), "binary", "outcomes"),
        ("roster", "visited", "1", "binary", "outcomes"),
    ],
)
def test_logs_require_binary_measurements_and_matching_key_domains(
    role, column, value, code, field
):
    frames = _frames()
    frames[role] = frames[role].astype(object)
    frames[role].at[2 if column == "visited" else 0, column] = value
    experiment, reader = _register(frames)
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == f"spillover.log.{code}"
    assert exc.value.context["field"] == field


@pytest.mark.parametrize("backend", ["arrow", "polars"])
def test_nan_in_an_unoffered_outcome_is_missing_even_without_an_arrow_null(backend):
    frames = _frames()
    roster = frames["roster"]
    values = roster["visited"].to_numpy(copy=True)
    values[2] = float("nan")
    arrow = pa.Table.from_pandas(roster.drop(columns="visited"), preserve_index=False)
    arrow = arrow.append_column("visited", pa.array(values, from_pandas=False))
    native = arrow if backend == "arrow" else pl.from_arrow(arrow)
    experiment, reader = _register(frames | {"roster": native})
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == "outcomes"


@pytest.mark.parametrize("field", ["group_policy", "offered"])
def test_both_realized_quotas_are_local_not_just_trial_totals(field):
    frames = _frames()
    roster = frames["roster"]
    roster["region"] = roster.store.map(lambda g: "west" if g in {10, 2, 12} else "east")
    design = _design(
        groups_per_policy={s: {"half": 1, "zero": 1, "full": 1} for s in ("east", "west")}
    )
    if field == "group_policy":
        frames[field].loc[[0, 4], "policy"] = ["zero", "half"]
    else:
        roster.loc[roster.customer == "person/10/0", "assignment"] = "no offer"
        roster.loc[roster.customer == "person/2/0", "assignment"] = "coupon"
    experiment, reader = _register(frames, design)
    profiles = experiment.analysis_context["planning_groups"]
    assert [(stratum, size) for stratum, size, _ in profiles] == [
        ("east", 6),
        ("east", 4),
        ("east", 5),
        ("west", 3),
        ("west", 4),
        ("west", 2),
    ]
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == "spillover.log.quota"
    assert exc.value.context["field"] == field


def test_stratum_totals_must_account_for_every_group():
    frames = _frames()
    design = _design(groups_per_policy={"all": {"zero": 2, "half": 2, "full": 1}})
    with pytest.raises(InvalidRequestError) as exc:
        _register(frames, design)
    assert exc.value.code == "spillover.design.quota"
    assert exc.value.context["field"] == "groups_per_policy"


def test_planning_needs_no_observed_frames_but_readout_requires_them():
    frames = _frames()
    roster = frames["roster"].drop(columns="region")
    planning = {"roster": roster[["customer", "store"]], "offer_counts": frames["offer_counts"]}
    design = _design(groups_per_policy={"midwest": {"zero": 2, "half": 2, "full": 2}})
    metric = MetricSpec(name="visited", type="conversion", window_days=7)
    experiment, reader = _register(planning, design, stratum=None, metric=metric)
    with reader, pytest.raises(InvalidRequestError) as exc:
        run(experiment, reader)
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == "group_policy"

    planning["group_policy"] = frames["group_policy"]
    for field, columns in (
        ("offered", ["customer", "store"]),
        ("outcomes", ["customer", "store", "assignment"]),
    ):
        planning["roster"] = roster[columns]
        _, reader = _register(planning, design, stratum=None, metric=metric)
        with reader, pytest.raises(InvalidRequestError) as missing:
            run(experiment, reader)
        assert missing.value.code == "spillover.log.keys"
        assert missing.value.context["field"] == field

    planning["roster"] = roster
    roster["visited"] = pd.array([0] * len(roster), dtype="Int64")
    _, reader = _register(planning, design, stratum=None, metric=metric)
    with reader:
        assert _effect(experiment, reader).estimate == 0
        roster.at[2, "visited"] = pd.NA
        with pytest.raises(InvalidRequestError) as unknown:
            run(experiment, reader)
        assert unknown.value.code == "spillover.log.keys"
        assert unknown.value.context["field"] == "outcomes"


def test_multiple_declared_strata_require_a_stratum_column():
    frames = _frames()
    design = _design(groups_per_policy={s: {"half": 1, "zero": 1, "full": 1} for s in ("x", "y")})
    with pytest.raises(InvalidRequestError) as ambiguous:
        _register(frames, design, stratum=None)
    assert ambiguous.value.code == "spillover.design.invalid"
    assert ambiguous.value.context["field"] == "stratum"


def test_changing_only_the_selected_outcome_column_invalidates_a_saved_registration():
    frames = _frames()
    original, reader = _register(frames)
    with reader:
        saved = original.to_json()
    # No measurement column yet: the mismatch must be detected before reading outcomes.
    metric = _METRIC.model_copy(update={"value_column": "visited_later"})
    _, rebound = _register(frames, metric=metric)
    with rebound, pytest.raises(CodedError) as stale:
        run(Experiment.from_json(saved), rebound)
    assert stale.value.code == "kernel.run.execution_unavailable"
