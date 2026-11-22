"""Binary untreated outcomes under fixed group-policy and within-group treatment counts."""

from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations, permutations, product
from math import prod, sqrt
from statistics import mean, variance
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from increment._spillover_design import GroupProfile, SpilloverPopulation
from increment.errors import InvalidRequestError
from increment.estimation import spillover
from increment.estimation.spillover import (
    BatchCell,
    estimate_comparison,
    reason_text,
    summarize_cells,
    summarize_counts,
)
from increment.spillover import SpilloverComparison, SpilloverDesign, from_frame


def _comparison(policy="upper", reference="base"):
    return SpilloverComparison(policy=policy, reference=reference, meaningful_effect=0.1)


def _group(i, size=4, offers=(1, 2), stratum="all"):
    return f"g{i:03d}", stratum, size, offers


def _population(groups, quotas=None, comparisons=None, *, alpha=None):
    design = SpilloverDesign(
        groups_per_policy=(
            {"all": {"base": len(groups) // 2, "upper": len(groups) // 2}}
            if quotas is None
            else quotas
        ),
        comparisons=(_comparison(),) if comparisons is None else comparisons,
        outcome="visit",
        horizon="7d",
        **({} if alpha is None else {"alpha": alpha}),
    )
    roster = pd.DataFrame(
        [(f"{g}/{j}", g, s) for g, s, n, _ in groups for j in range(n)],
        columns=["person_id", "group_id", "stratum"],
    )
    counts = pd.DataFrame(
        [(g, p, k) for g, _, _, ks in groups for p, k in zip(design.policies, ks, strict=True)],
        columns=["group_id", "policy", "offers"],
    )
    experiment, reader = from_frame(roster, offer_counts=counts, design=design)
    with reader:
        return experiment.population


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
        SimpleNamespace(
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


def test_subnormal_family_alpha_has_a_finite_nontrivial_limit():
    pop = _population(tuple(_group(i) for i in range(240)), alpha=np.nextafter(0.0, 1.0))
    labels = ["base"] * 120 + ["upper"] * 120
    cells = _cells(pop, labels, [0] * 120 + [2] * 120)
    lo, hi = spillover.SpilloverInterval(pop).bounds(_comparison(), cells)
    # Decimal inversion of mean_j(1-eta_j+eta_j/m)^120 = 4/alpha.
    assert lo[0] == pytest.approx(-0.9962671986146618, rel=0, abs=1e-12)
    assert hi[0] == 1


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
        result.available & (abs(result.estimate) > 1.959963984540054 * result.standard_error),
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


def test_bounded_batches_preserve_centering_and_cast_before_multiplication():
    # All nine copies of a fractional total are equal: mean rounding is not real spread.
    groups = tuple(_group(i, 120, (1, 7)) for i in range(18))
    pop = _population(groups)
    assignment = np.tile([0, 1] * 9, (16_384, 1))
    successes = np.full(assignment.shape, 73, dtype=np.int8)
    successes[1::3, 3] += 1
    successes[2::3, 0] -= 1
    result = summarize_counts(pop, _comparison(), assignment, successes)
    for i, row in enumerate(successes[:3]):
        point, se, reason = estimate_comparison(
            pop, _comparison(), _cells(pop, ["base", "upper"] * 9, row)
        )
        np.testing.assert_allclose(result.estimate[i::3], point, rtol=0, atol=1e-15)
        assert {reason_text(c) for c in np.unique(result.reason[i::3])} == {reason}
        if se is None:
            assert i == 0 and reason == "zero_variance"
            assert np.isnan(result.standard_error[i::3]).all()
        else:
            assert i != 0
            np.testing.assert_allclose(result.standard_error[i::3], se, rtol=1e-13, atol=0)


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


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (None, (-0.066319975405384, 0.120703247780779)),
        (0.01, (-0.083351528408934, 0.137919690345054)),
    ],
)
def test_fixed_mixture_240_store_counterexample(alpha, expected):
    """Four always-positive stores all land in the higher-offer policy under a true null."""
    pop = _population(tuple(_group(i, 59, (0, 1)) for i in range(240)), alpha=alpha)
    events = [14] * 120 + [58] * 4 + [13] * 28 + [14] * 88
    cells = _cells(pop, ["base"] * 120 + ["upper"] * 120, events)
    rule = spillover.SpilloverInterval(pop)
    lo, hi = rule.bounds(_comparison(), cells)
    assert lo.shape == hi.shape == (1,)
    assert (lo[0], hi[0]) == pytest.approx(expected, abs=1e-10, rel=0)
    # Label order and repeated planner assignments do not change the compressed law.
    codes = np.tile([0] * 120 + [1] * 120, (7, 1))
    dense = rule.bounds_from_counts(_comparison(), codes, np.tile(events, (7, 1)))
    np.testing.assert_allclose(dense, np.tile(np.array(expected)[:, None], 7), rtol=0, atol=1e-10)


def _decimal_tail(strata, weights, threshold):
    """Direct products from normalized group observations, not floating cell summaries."""
    counts = [len(xs) for xs in strata]
    rates = [w / n for w, n in zip(weights, counts, strict=True)]
    if not any(any(xs) for xs in strata):
        return Decimal(0)
    atoms = []
    for xs in strata:
        a, q = mean(xs), mean(x * x for x in xs)
        atoms.append((q / a, len(xs) * a**2 / q) if a else (Decimal(0), Decimal(0)))
    bets = [Decimal(".95") / 2**j for j in range(1 + (max(counts) - 1).bit_length())]
    etas = [[t * r / max(rates) for r in rates] for t in bets]

    def wealth(m, *, verify_compression=False):
        terms = []
        for eta in etas:
            term = Decimal(1)
            for e, xs, (b, count) in zip(eta, strata, atoms, strict=True):
                compressed = (1 - e) ** (len(xs) - count) * (1 - e + e * b / m) ** count
                if verify_compression:
                    raw = prod(1 - e + e * x / m for x in xs)
                    assert compressed <= raw * (1 + Decimal("1e-45"))
                term *= compressed
            terms.append(term)
        return mean(terms)

    wealth(Decimal(".31"), verify_compression=True)
    lo, hi = Decimal(0), Decimal(1)
    for _ in range(82):
        mid = (lo + hi) / 2
        if wealth(mid) >= threshold:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _decimal_policy(pop, labels, events, policy, k):
    expanded, xs = [], []
    for stratum in pop.design.groups_per_policy:
        groups = [
            (g, p, y)
            for g, p, y in zip(pop.groups, labels, events, strict=True)
            if g.stratum == stratum
        ]
        b = max(g.size for g, _, _ in groups)
        expanded.append(len(groups) * b)
        idx = pop.design.policies.index(policy)
        xs.append(
            [
                Decimal(g.size * y) / ((g.size - g.offers[idx]) * b)
                for g, p, y in groups
                if p == policy
            ]
        )
    weights = [Decimal(e) / sum(expanded) for e in expanded]
    scale = Decimal(sum(expanded)) / sum(g.size for g in pop.groups)
    threshold = 2 * k / Decimal.from_float(pop.design.alpha)
    lower = scale * _decimal_tail(xs, weights, threshold)
    upper = scale * (1 - _decimal_tail([[1 - x for x in row] for row in xs], weights, threshold))
    return min(lower, 1), min(upper, 1)


def test_compressed_decimal_roots_for_unequal_groups_shared_reference_and_nuisance():
    labels = (
        ["base", "medium", "spare", "upper"] * 6
        + ["base", "base", "spare", "medium"] * 2
        + ["upper", "medium", "base"] * 8
        + ["upper", "upper", "medium", "spare"] * 3
    )
    sizes = [7 + (i * 5) % (7 if i < 32 else 10) for i in range(len(labels))]
    groups = tuple(_group(i, n, (0, 2, 1, 4), "a" if i < 32 else "b") for i, n in enumerate(sizes))
    comparisons = (_comparison("medium"), _comparison("upper"))
    quotas = {"a": dict(Counter(labels[:32])), "b": dict(Counter(labels[32:]))}
    pop = _population(groups, quotas, comparisons, alpha=0.13)
    events = [
        (i * 7 + 3) % (n - (0, 2, 1, 4)[pop.design.policies.index(p)] + 1)
        for i, (n, p) in enumerate(zip(sizes, labels, strict=True))
    ]
    cells = _cells(pop, labels, events)
    rule = spillover.SpilloverInterval(pop)
    with localcontext() as ctx:
        ctx.prec = 58
        # Two pairs, three compared policies, four registered policy levels.
        expected = {
            p: _decimal_policy(pop, labels, events, p, 3) for p in ("base", "medium", "upper")
        }
        for c in comparisons:
            p, r = expected[c.policy], expected[c.reference]
            want = (p[0] - r[1], p[1] - r[0])
            codes = np.array([[{c.policy: 1, c.reference: 0}.get(label, 2) for label in labels]])
            projected = np.where(codes == 2, 0, events)
            for lo, hi in (rule.bounds(c, cells), rule.bounds_from_counts(c, codes, projected)):
                assert Decimal.from_float(lo[0]) <= want[0]
                assert Decimal.from_float(hi[0]) >= want[1]
                assert (lo[0], hi[0]) == pytest.approx(tuple(map(float, want)), rel=0, abs=1e-10)


def test_one_group_and_no_events_have_the_fixed_single_bet_limit():
    pop = _population((_group(0), _group(1)))
    lo, hi = spillover.SpilloverInterval(pop).bounds(
        _comparison(), _cells(pop, ["base", "upper"], [0, 0])
    )
    # All normalized complements are one. One group per policy implies j=0 only.
    upper = 1 - 0.95 / (4 / 0.05 - 0.05)
    assert lo[0] == pytest.approx(-upper, rel=0, abs=1e-13)
    assert hi[0] == pytest.approx(upper, rel=0, abs=1e-13)


def test_outside_domain_point_does_not_reverse_the_clipped_policy_range():
    groups = tuple(_group(i, 2 if i < 30 else 100, (0, 1)) for i in range(60))
    pop = _population(groups)
    cells = _cells(pop, ["base"] * 30 + ["upper"] * 30, [0] * 30 + [99] * 30)
    point = estimate_comparison(pop, _comparison(), cells)[0]
    assert point == pytest.approx(100 / 51)
    rule = spillover.SpilloverInterval(pop)
    lo, hi = rule.bounds(_comparison(), cells)
    with localcontext() as ctx:
        ctx.prec = 58
        m = _decimal_tail([[Decimal(1)] * 30], [Decimal(1)], Decimal(80))
        # Higher-offer policy's untruncated lower and upper limits both exceed one.
        scale = Decimal(100) / 51
        assert scale * m > 1
        assert lo[0] == pytest.approx(float(1 - scale * (1 - m)), rel=0, abs=1e-10)
    assert hi[0] == 1


@pytest.mark.parametrize(
    ("mean_total", "ss", "events"),
    [(np.inf, 0, 1), (1, np.nan, 1), (-1, 0, 0), (6, 0, 1), (0, 1, 0), (1, -0.1, 1), (1, 3, 1)],
)
def test_infeasible_policy_moments_are_not_intervals(mean_total, ss, events):
    # When two totals have mean one their largest possible centered SS is two.
    pop = _population(tuple(_group(i) for i in range(4)))
    good = BatchCell(np.zeros(1), np.zeros(1), np.array([0]), np.array([6]))
    bad = BatchCell(np.array([mean_total]), np.array([ss]), np.array([events]), np.array([4]))
    with pytest.raises(InvalidRequestError) as exc:
        spillover.SpilloverInterval(pop).bounds(
            _comparison(), {("all", "base"): good, ("all", "upper"): bad}
        )
    assert exc.value.code == "spillover.counts.invalid"


def _joint_stratum(pop, indices, potentials):
    """Enumerate both stages; different offers give assignments different probabilities."""
    policies = pop.design.policies
    quotas = pop.design.groups_per_policy[pop.groups[indices[0]].stratum]
    schedule = [j for j, p in enumerate(policies) for _ in range(quotas[p])]
    partitions = sorted(set(permutations(schedule)))
    states = []
    for labels in partitions:
        urns = []
        for i, p in zip(indices, labels, strict=True):
            g = pop.groups[i]
            urns.append(
                Counter(
                    sum(member < potentials[i][p] for member in subset)
                    for subset in combinations(range(g.size), g.size - g.offers[p])
                )
            )
        denominator = len(partitions) * prod(urn.total() for urn in urns)
        for events in product(*urns):
            ways = prod(urn[y] for urn, y in zip(urns, events, strict=True))
            states.append((labels, events, Fraction(ways, denominator)))
    assert sum(weight for _, _, weight in states) == 1
    return states


def _stratum_bet_expectation(pop, block, states, b, p, potentials, *, target, eta, reflected):
    def bet(x):
        ratio = (1 - x) / (1 - target) if reflected else x / target
        return 1 - eta + eta * ratio

    observed = Fraction(0)
    for labels, events, weight in states:
        factors = []
        for i, label, y in zip(block, labels, events, strict=True):
            if label == p:
                group = pop.groups[i]
                x = Fraction(group.size * y, (group.size - group.offers[p]) * b)
                factors.append(bet(x))
        observed += weight * prod(factors)

    # Averaging the second-stage draws replaces sampled totals by the potential events.
    expected = [bet(Fraction(potentials[i][p], b)) for i in block]
    draws = pop.design.groups_per_policy[pop.groups[block[0]].stratum][pop.design.policies[p]]
    assert observed == mean(prod(s, start=Fraction(1)) for s in combinations(expected, draws))
    return observed


def test_exact_two_stage_law_covers_the_person_targets_and_both_normalized_tails():
    sizes = (2, 3, 4, 2, 3, 2, 3)
    groups = tuple(
        _group(i, size, (0, 1, size - 1), "a" if i < 4 else "b") for i, size in enumerate(sizes)
    )
    comparisons = (_comparison("medium"), _comparison("upper"))
    policies = ("base", "medium", "upper")
    quotas = {"a": {"base": 2, "medium": 1, "upper": 1}, "b": dict.fromkeys(policies, 1)}
    pop = _population(groups, quotas, comparisons, alpha=0.25)
    # Each entry is the number of people with untreated outcome one under that
    # policy's fixed offer count. Where two offer counts coincide, outcomes agree.
    potentials = [(0, 2, 2), (2, 1, 2), (0, 4, 0), (2, 0, 0), (1, 2, 1), (0, 2, 2), (3, 0, 3)]
    blocks = (range(4), range(4, 7))
    strata = [_joint_stratum(pop, block, potentials) for block in blocks]
    maxima = [max(sizes[i] for i in block) for block in blocks]
    expanded = [len(block) * b for block, b in zip(blocks, maxima, strict=True)]
    people = sum(sizes)
    targets = [Fraction(sum(row[p] for row in potentials), people) for p in range(3)]
    # Direct product expectations at each policy's actual person target, not
    # an equally weighted group target or a complement on the person scale.
    for p, policy in enumerate(policies):
        ns = [q[policy] for q in quotas.values()]
        rates = [Fraction(e, n) for e, n in zip(expanded, ns, strict=True)]
        target = targets[p] * Fraction(people, sum(expanded))
        for reflected, j in product((False, True), range(1 + (max(ns) - 1).bit_length())):
            expectation = Fraction(1)
            for block, states, b, rate in zip(blocks, strata, maxima, rates, strict=True):
                eta = Fraction(19, 20 * 2**j) * rate / max(rates)
                expectation *= _stratum_bet_expectation(
                    pop,
                    block,
                    states,
                    b,
                    p,
                    potentials,
                    target=target,
                    eta=eta,
                    reflected=reflected,
                )
            assert expectation <= 1

    labels, events, weights = [], [], []
    for (a, y, pa), (b, z, pb) in product(*strata):
        labels.append(a + b)
        events.append(y + z)
        weights.append(pa * pb)
    assert len(set(weights)) > 1
    laws = np.array(labels)
    rule = spillover.SpilloverInterval(pop)
    missed = np.zeros(len(weights), dtype=bool)
    for c in comparisons:
        p, r = (pop.design.policies.index(name) for name in (c.policy, c.reference))
        codes = np.where(laws == p, 1, np.where(laws == r, 0, 2))
        counts = np.where(codes == 2, 0, events)
        truth = targets[p] - targets[r]
        points = summarize_counts(pop, c, codes, counts).estimate
        assert np.dot(np.array(weights, dtype=float), points) == pytest.approx(float(truth))
        lo, hi = rule.bounds_from_counts(c, codes, counts)
        missed |= (lo > float(truth)) | (hi < float(truth))
    assert sum(w for w, miss in zip(weights, missed, strict=True) if miss) <= Fraction(1, 4)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_many_one_group_strata_keep_the_outward_side_of_the_closed_form():
    # For every stratum and each policy, all observed normalized totals are
    # exactly 1/2. The product reduces to (.05 + .475/m)^8192 for either tail.
    n = 8192
    names = [f"s{i}" for i in range(n)]
    design = SpilloverDesign(
        {s: {"base": 1, "upper": 1} for s in names}, (_comparison(),), "visit", "7d"
    )
    groups = tuple(
        GroupProfile(2 * i + j, s, 4, (0, 2)) for i, s in enumerate(names) for j in range(2)
    )
    pop = SpilloverPopulation(design, groups, "many-strata")
    r = BatchCell(np.array([2.0]), np.zeros(1), np.array([2]), np.array([4]))
    p = BatchCell(np.array([2.0]), np.zeros(1), np.array([1]), np.array([2]))
    cells = {(s, policy): cell for s in names for policy, cell in (("base", r), ("upper", p))}
    lo, hi = spillover.SpilloverInterval(pop).bounds(_comparison(), cells)
    with localcontext() as ctx:
        ctx.prec = 58
        limit = Decimal(".475") / (Decimal(80) ** (Decimal(1) / n) - Decimal(".05"))
        assert Decimal.from_float(lo[0]) <= 2 * limit - 1
        assert Decimal.from_float(hi[0]) >= 1 - 2 * limit
        assert lo[0] == pytest.approx(float(2 * limit - 1), rel=0, abs=1e-10)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_long_rounded_cell_mean_encloses_exact_product_in_both_tails():
    n = 16_384
    design = SpilloverDesign({"all": {"base": n, "upper": n}}, (_comparison(),), "visit", "7d")
    groups = tuple(GroupProfile(i, "all", 10, (0, 7)) for i in range(2 * n))
    pop = SpilloverPopulation(design, groups, "constant-third")
    # Recorded DuckDB AVG of n copies of 10*1/3, with zero centered SS.
    # Reflect its signed error as well: 1-X must use the same absolute allowance.
    reported = float.fromhex("0x1.aaaaaaaaab0a4p+1")
    cells = {
        ("all", "base"): BatchCell(np.zeros(2), np.zeros(2), np.zeros(2, int), np.full(2, 10 * n)),
        ("all", "upper"): BatchCell(
            np.array([reported, 10 - reported]),
            np.zeros(2),
            np.array([n, 2 * n]),
            np.full(2, 3 * n),
        ),
    }
    lo, hi = spillover.SpilloverInterval(pop).bounds(_comparison(), cells)
    with localcontext() as ctx:
        ctx.prec = 58
        # With constant X=a, all roots are exactly a times the all-one root.
        bets = [Decimal(".95") / (2**j) for j in range(15)]
        left, right = Decimal(0), Decimal(1)
        for _ in range(90):
            mid = (left + right) / 2
            if sum((1 - t + t / mid) ** n for t in bets) >= 80 * len(bets):
                left = mid
            else:
                right = mid
        for i, a in enumerate((Decimal(1) / 3, Decimal(2) / 3)):
            lower, upper = (1 + a) * left - 1, 1 - (1 - a) * left
            assert Decimal.from_float(lo[i]) <= lower
            assert Decimal.from_float(hi[i]) >= upper
            assert (lo[i], hi[i]) == pytest.approx((float(lower), float(upper)), abs=1e-10, rel=0)
