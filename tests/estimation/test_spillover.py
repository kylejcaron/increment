"""Binary untreated outcomes under fixed group-policy and within-group treatment counts."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import combinations, product
from math import sqrt
from statistics import mean, variance
from typing import Any, cast

import numpy as np
import pytest
from scipy.special import ndtr

from increment._spillover_design import SpilloverDesign, SpilloverGroup
from increment.errors import InvalidRequestError
from increment.estimation.spillover import Z95, reason_text, summarize_counts


def _group(index, size=4, low=1, high=2, stratum="all"):
    return SpilloverGroup(
        str(index), tuple(f"{index}:{i}" for i in range(size)), low, high, stratum
    )


def _design(groups, high=None):
    return SpilloverDesign(
        groups, {"all": len(groups) // 2} if high is None else high, "visit", "7d"
    )


def _trial():
    groups = (_group("b"), _group("a"), _group("c"), _group("d"))
    high = {g.group_id: i % 2 for i, g in enumerate(groups)}
    treated = {
        person: i < (g.high_treated if high[g.group_id] else g.low_treated)
        for g in groups
        for i, person in enumerate(g.members)
    }
    outcomes = {person: int(person.endswith(":3")) for person in treated if not treated[person]}
    return _design(groups), high, treated, outcomes


def test_design_snapshots_labels_members_and_quota_in_canonical_order():
    people = ["amy", "bob", "cat"]
    a = SpilloverGroup("z", people, 1, 2, "A")
    groups = [_group("b", stratum="B"), a, _group("m", stratum="A"), _group("a", stratum="B")]
    quotas = {"B": np.int64(1), "A": 1}
    design = _design(groups, quotas)
    people.clear()
    groups.clear()
    quotas.clear()
    assert design.groups[1].members == ("amy", "bob", "cat")
    assert [(g.stratum, g.group_id) for g in design.groups] == [
        ("A", "m"),
        ("A", "z"),
        ("B", "a"),
        ("B", "b"),
    ]
    assert dict(design.high_groups) == {"A": 1, "B": 1}
    with pytest.raises(TypeError):
        cast(Any, design.high_groups)["A"] = 0
    with pytest.raises(FrozenInstanceError):
        cast(Any, a).stratum = "B"


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"group_id": " "}, "spillover.design.invalid"),
        ({"stratum": None}, "spillover.design.invalid"),
        ({"members": "abcd"}, "spillover.design.invalid"),
        ({"members": ("", "b", "c")}, "spillover.design.invalid"),
        ({"members": ("a", "b", "a")}, "spillover.design.duplicate"),
        ({"low_treated": True}, "spillover.design.quota"),
        ({"high_treated": 2.0}, "spillover.design.quota"),
        ({"high_treated": np.timedelta64(2, "ns")}, "spillover.design.quota"),
        ({"low_treated": 0}, "spillover.design.quota"),
        ({"low_treated": 2}, "spillover.design.quota"),
        ({"high_treated": 4}, "spillover.design.quota"),
    ],
)
def test_group_requires_identified_integer_laws_and_distinct_members(changes, code):
    with pytest.raises(InvalidRequestError) as exc:
        replace(_group(0), **changes)
    assert exc.value.code == code


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"outcome": ""}, "spillover.design.invalid"),
        ({"horizon": None}, "spillover.design.invalid"),
        ({"groups": ()}, "spillover.design.invalid"),
        ({"groups": ("a", "b")}, "spillover.design.invalid"),
        ({"high_groups": {}}, "spillover.design.quota"),
        ({"high_groups": {"all": 1, "other": 1}}, "spillover.design.quota"),
        ({"high_groups": {"all": True}}, "spillover.design.quota"),
        ({"high_groups": {"all": 1.0}}, "spillover.design.quota"),
        ({"high_groups": {"all": 0}}, "spillover.design.quota"),
        ({"high_groups": {"all": 2}}, "spillover.design.quota"),
    ],
)
def test_design_refuses_unidentified_first_stage_or_unspecified_measure(changes, code):
    with pytest.raises(InvalidRequestError) as exc:
        replace(_design((_group(0), _group(1))), **changes)
    assert exc.value.code == code


def test_even_different_strata_cannot_reuse_group_or_person_ids():
    a = _group("a")
    b = _group("b", stratum="other")
    for other in (replace(b, group_id="a"), replace(b, members=(*b.members[:-1], a.members[0]))):
        with pytest.raises(InvalidRequestError) as exc:
            _design((a, other, _group("c"), _group("d", stratum="other")), {"all": 1, "other": 1})
        assert exc.value.code == "spillover.design.duplicate"


def test_real_logs_match_ids_not_iteration_order_and_treated_outcomes_are_optional():
    design, high, treated, outcomes = _trial()
    outcomes["b:3"] = 0
    assigned, counts = design.logged_counts(high, treated, outcomes)
    assert assigned.dtype == np.bool_
    np.testing.assert_array_equal(assigned, [[True, False, False, True]])
    np.testing.assert_array_equal(counts, [[1, 0, 1, 1]])
    all_outcomes = {**dict.fromkeys(treated, np.int8(1)), **outcomes}
    reordered = dict(reversed(tuple(all_outcomes.items())))
    np.testing.assert_array_equal(design.logged_counts(high, treated, reordered)[1], counts)


@pytest.mark.parametrize("field", ["group_high", "treated", "outcomes"])
@pytest.mark.parametrize("mismatch", ["missing", "extra", "replaced"])
def test_real_logs_do_not_fill_missing_ids_or_drop_foreign_ids(field, mismatch):
    design, group_high, treated, outcomes = _trial()
    logs = {"group_high": group_high, "treated": treated, "outcomes": outcomes}
    record = logs[field]
    if mismatch != "extra":
        del record[next(iter(record))]
    if mismatch != "missing":
        record["foreign"] = 0
    with pytest.raises(InvalidRequestError) as exc:
        design.logged_counts(**logs)
    assert exc.value.code == "spillover.log.keys"
    assert exc.value.context["field"] == field


@pytest.mark.parametrize(
    ("field", "key", "value"),
    [
        ("group_high", "a", "1"),
        ("group_high", "b", np.float64(0.0)),
        ("treated", "a:0", 2),
        ("treated", "b:0", -1),
        ("outcomes", "b:3", np.timedelta64(1, "ns")),
        ("outcomes", "b:3", None),
        (
            "outcomes",
            "b:0",
            1.0,
        ),  # Optional treated measurements still belong to a binary endpoint.
    ],
)
def test_log_values_are_binary_without_truthiness_or_rounding(field, key, value):
    design, group_high, treated, outcomes = _trial()
    logs = {"group_high": group_high, "treated": treated, "outcomes": outcomes}
    logs[field][key] = value
    with pytest.raises(InvalidRequestError) as exc:
        design.logged_counts(**logs)
    assert exc.value.code == "spillover.log.binary"
    assert exc.value.context["field"] == field


def test_both_realized_quotas_are_local_not_just_roster_totals():
    groups = tuple(_group(i, stratum="left" if i < 2 else "right") for i in range(4))
    design = _design(groups, {"left": 1, "right": 1})
    for high, treated_counts, field in (
        ((1, 1, 0, 0), (2, 2, 1, 1), "group_high"),
        ((1, 0, 1, 0), (1, 2, 2, 1), "treated"),
    ):
        group_high = {g.group_id: v for g, v in zip(groups, high, strict=True)}
        treated = {
            p: j < n
            for g, n in zip(groups, treated_counts, strict=True)
            for j, p in enumerate(g.members)
        }
        with pytest.raises(InvalidRequestError) as exc:
            design.logged_counts(group_high, treated, dict.fromkeys(treated, 0))
        assert exc.value.code == "spillover.log.quota"
        assert exc.value.context["field"] == field


def test_rct2_unequal_size_person_weighted_untreated_contrast():
    """RCT2 CalAPO.R at 5820e34f: scale by size/mean size; contrast high(0) minus low(0)."""
    sizes = [4, 5, 6, 8, 7, 9]
    high = np.array([[True, False, True, False, False, True]])
    low = [1, 2, 2, 3, 2, 4]
    high_quota = [3, 3, 4, 5, 4, 6]
    groups = tuple(_group(i, n, low[i], high_quota[i]) for i, n in enumerate(sizes))
    successes = np.array([[1, 1, 0, 2, 4, 1]])
    result = summarize_counts(_design(groups), high, successes)
    assert result.available.tolist() == [True]
    assert reason_text(result.reason[0]) is None
    assert result.estimate[0] == pytest.approx(-0.17777777777777776, abs=1e-14)
    assert result.standard_error[0] ** 2 == pytest.approx(0.0651968734020016, abs=1e-14)
    assert ndtr(-Z95) == pytest.approx(0.025, abs=1e-15)


def test_enumerated_heterogeneous_null_has_twelve_false_calls_in_sixty_assignments():
    # Within each group everybody has the same outcome; stage two introduces no sampling noise.
    sizes = (4,) * 4 + (6,) * 5
    low = (0, 1, 0, 1, 0, 1, 0, 1, 0)
    high = (1, 0, *low[2:])
    assert sum(n * (h - l) for n, h, l in zip(sizes, high, low, strict=True)) == 0
    groups = tuple(_group(i, n, stratum="a" if i < 4 else "b") for i, n in enumerate(sizes))
    design = _design(groups, {"a": 2, "b": 3})
    policies, counts, points, variances = [], [], [], []
    for a, b in product(combinations(range(4), 2), combinations(range(4, 9), 3)):
        high_ids = {*a, *b}
        policy = [i in high_ids for i in range(9)]
        values = [high[i] if h else low[i] for i, h in enumerate(policy)]
        point = var = Fraction(0)
        for indices in (range(4), range(4, 9)):
            people_weight = Fraction(sum(sizes[i] for i in indices), sum(sizes))
            for arm, sign in ((True, 1), (False, -1)):
                arm_values = [Fraction(values[i]) for i in indices if policy[i] == arm]
                point += sign * people_weight * mean(arm_values)
                var += people_weight**2 * variance(arm_values) / len(arm_values)
        policies.append(policy)
        counts.append(
            [v * (n - (2 if h else 1)) for v, n, h in zip(values, sizes, policy, strict=True)]
        )
        points.append(float(point))
        variances.append(float(var))
    false_calls = [
        abs(p) > 1.959963984540054 * sqrt(v) for p, v in zip(points, variances, strict=True)
    ]
    assert (sum(false_calls), len(false_calls)) == (12, 60)
    result = summarize_counts(design, np.array(policies), np.array(counts))
    np.testing.assert_allclose(result.estimate, points, atol=1e-15)
    np.testing.assert_allclose(result.standard_error**2, variances, atol=1e-15)
    np.testing.assert_array_equal(
        result.available & (abs(result.estimate) > Z95 * result.standard_error), false_calls
    )


def test_absent_events_unit_variation_and_independent_group_variation_are_distinct():
    design = _design(tuple(_group(i, size=5) for i in range(6)))
    high = np.tile([True] * 3 + [False] * 3, (4, 1))
    successes = np.array([[0] * 6, [3] * 3 + [4] * 3, [1] * 3 + [2] * 3, [0, 0, 1, 0, 0, 0]])
    result = summarize_counts(design, high, successes)
    assert [reason_text(c) for c in result.reason] == [
        "no_outcome_variation",
        "no_outcome_variation",
        "zero_variance",
        None,
    ]
    assert result.reason.dtype.itemsize == 1
    np.testing.assert_array_equal(result.available, [False, False, False, True])
    assert np.isfinite(result.estimate).all()
    assert np.isnan(result.standard_error[:3]).all()
    assert result.standard_error[3] > 0


def test_group_size_variation_is_not_binary_outcome_variation():
    sizes = [3, 4, 8, 9]
    design = _design(tuple(_group(i, size=n) for i, n in enumerate(sizes)))
    result = summarize_counts(
        design, np.array([[True, True, False, False]]), np.array([[1, 2, 7, 8]])
    )
    assert result.estimate[0] == pytest.approx((3.5 - 8.5) * 4 / sum(sizes))
    assert reason_text(result.reason[0]) == "no_outcome_variation"
    assert np.isnan(result.standard_error[0])


def test_one_group_in_an_arm_still_identifies_point_and_takes_reason_precedence():
    design = _design((_group(0, size=3), _group(1, size=9)))
    high = np.array([[True, False]])
    result = summarize_counts(design, high, np.array([[1, 8]]))
    # All untreated outcomes are one; unequal group sizes still give different estimated totals.
    assert result.estimate[0] == pytest.approx(-1.0)
    assert reason_text(result.reason[0]) == "too_few_groups"
    assert not result.available[0] and np.isnan(result.standard_error[0])


@pytest.mark.parametrize(
    ("high", "successes"),
    [
        ([True, False], [[1, 0]]),
        ([[True, False]], [1, 0]),
        ([[1, 0]], [[1, 0]]),
        ([[True]], [[1]]),
        ([[True, True]], [[1, 0]]),
        ([[True, False]], [[1.25, 0]]),
        ([[True, False]], [[3, 0]]),
        ([[True, False]], [[0, -1]]),
    ],
)
def test_impossible_count_arrays_are_not_an_unavailable_estimate(high: Any, successes: Any):
    with pytest.raises(InvalidRequestError) as exc:
        summarize_counts(_design((_group(0), _group(1))), np.array(high), np.array(successes))
    assert exc.value.code == "spillover.counts.invalid"
