"""Binary untreated outcomes under fixed group-policy and within-group treatment counts."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
from math import sqrt
from statistics import mean, variance
from typing import Any

import numpy as np
import pytest
from scipy.special import ndtr

from increment._spillover_design import (
    GroupProfile,
    PolicyCell,
    SpilloverComparison,
    SpilloverDesign,
    SpilloverPopulation,
)
from increment.errors import InvalidRequestError
from increment.estimation import spillover as estimation
from increment.estimation.spillover import (
    BatchCell,
    critical_value,
    estimate_comparison,
    reason_text,
    summarize_cells,
    summarize_counts,
)


def _comparison(policy="upper", reference="base"):
    return SpilloverComparison(policy=policy, reference=reference, meaningful_effect=0.1)


def _group(i, size=4, offers=(1, 2), stratum="all"):
    return GroupProfile(group_id=f"g{i:03d}", stratum=stratum, size=size, offers=offers)


def _population(groups, quotas=None, comparisons=None):
    design = SpilloverDesign(
        groups_per_policy=(
            {"all": {"base": len(groups) // 2, "upper": len(groups) // 2}}
            if quotas is None
            else quotas
        ),
        comparisons=(_comparison(),) if comparisons is None else comparisons,
        outcome="visit",
        horizon="7d",
    )
    return SpilloverPopulation(design=design, groups=groups, roster_digest="a" * 64)


def _cells(population, labels, events):
    # Independent cell reduction in rational arithmetic, in arbitrary evidence order.
    totals = defaultdict(list)
    counts = defaultdict(int)
    unoffered = defaultdict(int)
    for g, label, e in zip(population.groups, labels, events, strict=True):
        untreated = g.size - g.offers[population.design.policies.index(label)]
        if untreated:
            key = (g.stratum, label)
            totals[key].append(Fraction(g.size * int(e), untreated))
            counts[key] += int(e)
            unoffered[key] += untreated
    return tuple(
        PolicyCell(
            stratum=s,
            policy=p,
            groups=len(ts),
            mean_total=float(mean(ts)),
            ss=float(sum((t - mean(ts)) ** 2 for t in ts)),
            unoffered=unoffered[s, p],
            events=counts[s, p],
        )
        for (s, p), ts in reversed(totals.items())
    )


def _check_rct2(population, labels, events, comparison, point, variance_):
    codes = np.array(
        [[{comparison.reference: 0, comparison.policy: 1}.get(label, 2) for label in labels]],
        dtype=np.int8,
    )
    result = summarize_counts(population, comparison, codes, np.where(codes == 2, 0, events))
    assert result.available.tolist() == [True]
    assert result.estimate[0] == pytest.approx(point, rel=0, abs=1e-14)
    assert result.standard_error[0] ** 2 == pytest.approx(variance_, rel=0, abs=1e-14)
    estimate, se, reason = estimate_comparison(
        population, comparison, _cells(population, labels, events)
    )
    assert reason is None and se is not None
    assert estimate == pytest.approx(point, rel=0, abs=1e-14)
    assert se**2 == pytest.approx(variance_, rel=0, abs=1e-14)


def test_rct2_unequal_size_person_weighted_untreated_contrast():
    """RCT2 CalAPO.R at 5820e34f: scale by size/mean size; contrast high(0) minus low(0)."""
    sizes = [4, 5, 6, 8, 7, 9]
    labels = ["upper", "base", "upper", "base", "base", "upper"]
    low = [1, 2, 2, 3, 2, 4]
    high_quota = [3, 3, 4, 5, 4, 6]
    pop = _population(tuple(_group(i, n, (low[i], high_quota[i])) for i, n in enumerate(sizes)))
    _check_rct2(
        pop, labels, [1, 1, 0, 2, 4, 1], _comparison(), -0.17777777777777776, 0.0651968734020016
    )


def test_rct2_three_policy_unequal_groups_shared_reference():
    # All six groups enter CalAPO; the second contrast is nonadjacent in its APO vector.
    sizes = (6, 8, 5, 7, 9, 10)
    groups = tuple(_group(i, n, (1, 2, 3)) for i, n in enumerate(sizes))
    comparisons = (_comparison("medium"), _comparison("upper"))
    pop = _population(groups, {"all": {"base": 2, "medium": 2, "upper": 2}}, comparisons)
    labels = ["base", "medium", "upper"] * 2
    events = [0, 2, 2, 3, 1, 4]
    _check_rct2(pop, labels, events, comparisons[0], 0.03015873015873019, 0.062920131015369099)
    _check_rct2(pop, labels, events, comparisons[1], 0.480952380952380909, 0.056712018140589557)


@pytest.mark.parametrize("reverse", [False, True])
def test_stratified_weights_include_both_saturated_nuisance_levels(reverse):
    sizes = (4, 6, 5, 7, 8, 9, 3, 10, 6, 8, 2, 7, 8, 9)
    groups = tuple(_group(i, n, (0, n, n, 1), "a" if i < 6 else "b") for i, n in enumerate(sizes))
    quotas = {
        "a": {"base": 2, "spare1": 1, "spare2": 1, "upper": 2},
        "b": {"base": 2, "spare1": 2, "spare2": 2, "upper": 2},
    }
    comparison = _comparison("base", "upper") if reverse else _comparison()
    pop = _population(groups, quotas, (comparison,))
    labels = ["base", "base", "upper", "upper", "spare1", "spare2"] * 2 + ["spare1", "spare2"]
    events = [1, 2, 4, 3, 0, 0, 2, 6, 5, 0, 0, 0, 0, 0]
    # HT expansion: group differences in A and B are 11/4 and -1. Conditioning on
    # the 49 people in eight realized comparison groups would instead give 7/49.
    point = Fraction(17, 184)
    var = Fraction(3445, 33856)
    _check_rct2(pop, labels, events, comparison, float(-point if reverse else point), float(var))


@pytest.mark.parametrize("family_size", [1, 2, 3, 10**18])
def test_critical_value_inverts_the_family_upper_tail(family_size):
    assert ndtr(-critical_value(family_size)) == pytest.approx(
        0.025 / family_size, rel=2e-14, abs=0
    )


def test_enumerated_heterogeneous_null_has_twelve_false_calls_in_sixty_assignments():
    # Within each group everybody has the same outcome; stage two introduces no sampling noise.
    sizes = (4,) * 4 + (6,) * 5
    low = (0, 1, 0, 1, 0, 1, 0, 1, 0)
    high = (1, 0, *low[2:])
    assert sum(n * (h - l) for n, h, l in zip(sizes, high, low, strict=True)) == 0
    groups = tuple(_group(i, n, stratum="a" if i < 4 else "b") for i, n in enumerate(sizes))
    pop = _population(groups, {"a": {"base": 2, "upper": 2}, "b": {"base": 2, "upper": 3}})
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
    result = summarize_counts(
        pop, _comparison(), np.array(policies, dtype=np.uint8), np.array(counts)
    )
    np.testing.assert_allclose(result.estimate, points, atol=1e-15)
    np.testing.assert_allclose(result.standard_error**2, variances, atol=1e-15)
    np.testing.assert_array_equal(
        result.available & (abs(result.estimate) > critical_value(1) * result.standard_error),
        false_calls,
    )


def test_absent_events_unit_variation_and_independent_group_variation_are_distinct():
    pop = _population(tuple(_group(i, size=5) for i in range(6)))
    assignment = np.tile([1] * 3 + [0] * 3, (4, 1))
    successes = np.array([[0] * 6, [3] * 3 + [4] * 3, [1] * 3 + [2] * 3, [0, 0, 1, 0, 0, 0]])
    result = summarize_counts(pop, _comparison(), assignment, successes)
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
    for i, row in enumerate(successes):
        point, se, reason = estimate_comparison(
            pop, _comparison(), _cells(pop, ["upper"] * 3 + ["base"] * 3, row)
        )
        assert point == pytest.approx(result.estimate[i])
        assert reason == reason_text(result.reason[i])
        assert (se is not None) == result.available[i]


def test_outcome_variation_only_in_another_comparison_cannot_rescue_interval():
    pop = _population(
        tuple(_group(i, 6, (0, 1, 2)) for i in range(6)),
        {"all": {"base": 2, "medium": 2, "upper": 2}},
        (_comparison("medium"), _comparison()),
    )
    cells = _cells(pop, ["base", "medium", "upper"] * 2, [0, 0, 2, 0, 0, 4])
    assert estimate_comparison(pop, _comparison("medium"), cells) == (
        0.0,
        None,
        "no_outcome_variation",
    )
    assert estimate_comparison(pop, _comparison(), cells)[2] is None


def test_projected_cell_batches_retain_point_for_nonfinite_variance():
    pop = _population(tuple(_group(i) for i in range(4)))
    reference = BatchCell(
        mean_total=np.zeros(3), ss=np.array([0.0]), events=np.array([0]), unoffered=np.array([6])
    )
    policy = BatchCell(
        mean_total=np.ones(3),
        ss=np.array([2.0, float("inf"), float("nan")]),
        events=np.array([1]),
        unoffered=np.array([4]),
    )
    result = summarize_cells(
        pop, _comparison(), {("all", "base"): reference, ("all", "upper"): policy}
    )
    np.testing.assert_array_equal(result.estimate, [0.25] * 3)
    np.testing.assert_allclose(result.standard_error, [0.25, np.nan, np.nan])
    assert [reason_text(c) for c in result.reason] == [
        None,
        "nonfinite_variance",
        "nonfinite_variance",
    ]


def test_bounded_batches_preserve_centering_and_cast_before_multiplication(monkeypatch):
    # All nine copies of a fractional total are equal: mean rounding is not real spread.
    groups = tuple(_group(i, 120, (1, 7)) for i in range(18))
    pop = _population(groups)
    assignment = np.tile([0, 1] * 9, (3, 1))
    successes = np.full(assignment.shape, 73, dtype=np.int8)
    successes[1, 3] += 1
    successes[2, 0] -= 1
    monkeypatch.setattr(estimation, "_WORK_CELLS", 18)
    result = summarize_counts(pop, _comparison(), assignment, successes)
    for i, row in enumerate(successes):
        point, se, reason = estimate_comparison(
            pop, _comparison(), _cells(pop, ["base", "upper"] * 9, row)
        )
        assert result.estimate[i] == pytest.approx(point, rel=0, abs=1e-15)
        assert reason_text(result.reason[i]) == reason
        if se is None:
            assert i == 0 and reason == "zero_variance" and np.isnan(result.standard_error[i])
        else:
            assert i != 0
            assert result.standard_error[i] == pytest.approx(se, rel=1e-13)


def test_group_size_and_nuisance_observations_are_not_comparison_outcome_variation():
    sizes = [3, 4, 8, 9, 6]
    pop = _population(
        tuple(_group(i, n, (1, 1, 2)) for i, n in enumerate(sizes)),
        {"all": {"base": 2, "spare": 1, "upper": 2}},
    )
    # Every compared unoffered person has the event. The different sizes create
    # variation in T; the nuisance cell has both outcomes. Neither licenses an SE.
    result = summarize_counts(
        pop, _comparison(), np.array([[1, 1, 0, 0, 2]]), np.array([[1, 2, 7, 8, 0]])
    )
    point = (3.5 - 8.5) * 5 / sum(sizes)
    assert result.estimate[0] == pytest.approx(point)
    assert reason_text(result.reason[0]) == "no_outcome_variation"
    assert np.isnan(result.standard_error[0])
    cells = _cells(pop, ["upper", "upper", "base", "base", "spare"], [1, 2, 7, 8, 3])
    observed, se, reason = estimate_comparison(pop, _comparison(), cells)
    assert observed == pytest.approx(point)
    assert se is None and reason == "no_outcome_variation"


def test_one_group_in_an_arm_still_identifies_point_and_takes_reason_precedence():
    pop = _population((_group(0, size=3), _group(1, size=9)))
    result = summarize_counts(pop, _comparison(), np.array([[1, 0]]), np.array([[1, 8]]))
    # All untreated outcomes are one; unequal group sizes still give different estimated totals.
    assert result.estimate[0] == pytest.approx(-1.0)
    assert reason_text(result.reason[0]) == "too_few_groups"
    assert not result.available[0] and np.isnan(result.standard_error[0])
    assert estimate_comparison(pop, _comparison(), _cells(pop, ["upper", "base"], [1, 8])) == (
        -1.0,
        None,
        "too_few_groups",
    )


@pytest.mark.parametrize(
    ("assignment", "successes"),
    [
        ([1, 0], [[1, 0]]),
        ([[1, 0]], [1, 0]),
        ([[True, False]], [[1, 0]]),
        ([[1.0, 0.0]], [[1, 0]]),
        ([[1]], [[1]]),
        ([[1, 1]], [[1, 0]]),
        ([[1, 2]], [[1, 0]]),
        ([[-1, 0]], [[1, 0]]),
        ([[3, 0]], [[1, 0]]),
        ([[1, 0]], [[1.25, 0]]),
        ([[1, 0]], [[3, 0]]),
        ([[1, 0]], [[0, -1]]),
    ],
)
def test_impossible_count_arrays_are_not_an_unavailable_estimate(assignment: Any, successes: Any):
    with pytest.raises(InvalidRequestError) as exc:
        summarize_counts(
            _population((_group(0), _group(1))),
            _comparison(),
            np.array(assignment),
            np.array(successes),
        )
    assert exc.value.code == "spillover.counts.invalid"


def test_projected_assignments_require_local_quotas_and_zero_other_successes():
    groups = tuple(_group(i, 6, (0, 1, 2), "a" if i < 3 else "b") for i in range(6))
    pop = _population(groups, {s: {"base": 1, "medium": 1, "upper": 1} for s in ("a", "b")})
    for assignment, counts in (
        ([1, 0, 2, 1, 0, 2], [0, 0, 1, 0, 0, 0]),
        ([1, 1, 2, 0, 0, 2], [0] * 6),
    ):
        with pytest.raises(InvalidRequestError) as exc:
            summarize_counts(pop, _comparison(), np.array([assignment]), np.array([counts]))
        assert exc.value.code == "spillover.counts.invalid"
